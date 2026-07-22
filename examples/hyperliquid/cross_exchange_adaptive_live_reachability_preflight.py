#!/usr/bin/env python3
"""Build source-materialized adaptive-pricing reachability evidence."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0722T064"
REPAIR_TASK_ID = "0722T065"
SCHEMA_VERSION = "cross_exchange_adaptive_live_reachability_v1"
RECEIPT_SCHEMA_VERSION = "adaptive_source_receipt_v1"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "adaptive_live_evidence_reachability_0722T064"
)
DEFAULT_RECEIPT_INDEX_PATH = DEFAULT_OUTPUT_DIR / "source_receipt_index.json"
DEFAULT_WATCHER_PATH = (
    PROJECT_ROOT
    / "examples"
    / "hyperliquid"
    / "hyperliquid_tiny_live_m2_public_watcher.py"
)
EXPECTED_SOURCE_FILENAMES = [
    "online_estimator_snapshot.json",
    "online_intensity_fit.csv",
    "quote_exposure_intervals.csv",
    "fill_feedback_snapshot.json",
    "fill_feedback_lifecycle_matrix.csv",
]
EXPECTED_SOURCE_TASK_IDS = {"0721T047", "0722T052"}
MIN_DYNAMIC_OBSERVATIONS_PER_SIDE = 3
MIN_DYNAMIC_DISTANCE_VARIATIONS_PER_SIDE = 2
MIN_FILL_FEEDBACK_OBSERVATIONS = 5
MIN_FILL_FEEDBACK_EXPOSURE_SECONDS = 25.0
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ReachabilityError(ValueError):
    """Raised when receipt, source bytes, or derived evidence is invalid."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReachabilityError(f"json_object_required:{path.name}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _canonical_hash(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        raise ReachabilityError("artifact_path_outside_project") from None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _csv_rows(data: bytes, *, filename: str) -> list[dict[str, str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReachabilityError(f"source_file_not_utf8:{filename}") from exc
    return list(csv.DictReader(io.StringIO(text)))


def _source_timeline(watcher_path: Path) -> dict[str, Any]:
    text = watcher_path.read_text(encoding="utf-8")
    function_token = "def run_event_driven_inline_reprice_live("
    start = text.find(function_token)
    if start < 0:
        raise ReachabilityError("watcher_inline_reprice_function_missing")
    section = text[start:]
    tokens = {
        "manager_cycle_call": "task7_manager_cycle = run_task7_manager_cycle(",
        "dynamic_candidate_argument": "dynamic_spread_candidate=(",
        "fill_feedback_candidate_argument": "fill_feedback_candidate=(",
        "dynamic_exposure_observation": (
            "state.online_estimator.observe_quote_exposure("
        ),
        "fill_feedback_finalization": (
            "feedback_artifacts = write_fill_feedback_artifacts("
        ),
    }
    positions: dict[str, int] = {}
    lines: dict[str, int] = {}
    for name, token in tokens.items():
        position = section.find(token)
        if position < 0:
            raise ReachabilityError(f"watcher_timeline_token_missing:{name}")
        positions[name] = position
        lines[name] = text.count("\n", 0, start + position) + 1
    if not (
        positions["manager_cycle_call"]
        < positions["dynamic_candidate_argument"]
        < positions["dynamic_exposure_observation"]
        and positions["manager_cycle_call"]
        < positions["fill_feedback_candidate_argument"]
        < positions["fill_feedback_finalization"]
    ):
        raise ReachabilityError("watcher_timeline_order_mismatch")
    return {
        "watcher_path": _portable_path(watcher_path),
        "watcher_sha256": _file_hash(watcher_path),
        "manager_cycle_call_count": section.count(
            tokens["manager_cycle_call"]
        ),
        "line_numbers": lines,
        "dynamic_candidate_is_manager_call_argument": True,
        "dynamic_evidence_observed_after_manager_returns": True,
        "fill_feedback_candidate_is_manager_call_argument": True,
        "fill_feedback_lifecycle_built_after_loop_finalization": True,
    }


def _validate_receipt_index(index: dict[str, Any]) -> list[dict[str, str]]:
    if index.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ReachabilityError("receipt_index_schema_mismatch")
    if index.get("task_id") != REPAIR_TASK_ID:
        raise ReachabilityError("receipt_index_task_id_mismatch")
    instance_id = str(index.get("instance_id") or "")
    if not re.fullmatch(r"i-[0-9a-f]+", instance_id):
        raise ReachabilityError("receipt_index_instance_id_invalid")
    packages = index.get("packages")
    if not isinstance(packages, list) or len(packages) != 2:
        raise ReachabilityError("receipt_index_packages_mismatch")
    output: list[dict[str, str]] = []
    task_ids: set[str] = set()
    for package in packages:
        if not isinstance(package, dict) or set(package) != {
            "task_id",
            "command_id",
            "remote_root",
            "receipt_path",
        }:
            raise ReachabilityError("receipt_index_package_fields_mismatch")
        command_id = str(package["command_id"])
        if not re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            command_id,
        ):
            raise ReachabilityError("receipt_index_command_id_invalid")
        receipt_path = PROJECT_ROOT / str(package["receipt_path"])
        try:
            receipt_path.resolve().relative_to(PROJECT_ROOT)
        except ValueError:
            raise ReachabilityError(
                "receipt_index_path_outside_project"
            ) from None
        task_id = str(package["task_id"])
        if task_id in task_ids:
            raise ReachabilityError("receipt_index_duplicate_task_id")
        task_ids.add(task_id)
        output.append(
            {
                **{key: str(value) for key, value in package.items()},
                "instance_id": instance_id,
                "receipt_path": str(receipt_path),
            }
        )
    if task_ids != EXPECTED_SOURCE_TASK_IDS:
        raise ReachabilityError("receipt_index_source_tasks_mismatch")
    return output


def _decode_receipt(
    package: dict[str, str],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    receipt_path = Path(package["receipt_path"])
    receipt = _read_json(receipt_path)
    if receipt.get("Status") != "Success" or receipt.get("ResponseCode") != 0:
        raise ReachabilityError("ssm_invocation_not_success")
    if receipt.get("CommandId") != package["command_id"]:
        raise ReachabilityError("ssm_invocation_command_id_mismatch")
    if receipt.get("InstanceId") != package["instance_id"]:
        raise ReachabilityError("ssm_invocation_instance_id_mismatch")
    if str(receipt.get("StandardErrorContent") or ""):
        raise ReachabilityError("ssm_invocation_stderr_not_empty")
    stdout = str(receipt.get("StandardOutputContent") or "").strip()
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ReachabilityError("ssm_invocation_stdout_not_json") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "task_id",
        "remote_root",
        "files",
    }:
        raise ReachabilityError("source_receipt_fields_mismatch")
    if payload["schema_version"] != RECEIPT_SCHEMA_VERSION:
        raise ReachabilityError("source_receipt_schema_mismatch")
    if payload["task_id"] != package["task_id"]:
        raise ReachabilityError("source_receipt_task_id_mismatch")
    if payload["remote_root"] != package["remote_root"]:
        raise ReachabilityError("source_receipt_remote_root_mismatch")
    files = payload["files"]
    if not isinstance(files, dict) or set(files) != set(
        EXPECTED_SOURCE_FILENAMES
    ):
        raise ReachabilityError("source_receipt_filenames_mismatch")
    materialized_dir = output_dir / "source_files" / package["task_id"]
    source_bytes: dict[str, bytes] = {}
    file_hashes: dict[str, str] = {}
    materialized_paths: dict[str, str] = {}
    for filename in EXPECTED_SOURCE_FILENAMES:
        entry = files[filename]
        if not isinstance(entry, dict) or set(entry) != {
            "sha256",
            "content_base64",
        }:
            raise ReachabilityError(f"source_receipt_file_fields:{filename}")
        remote_hash = str(entry["sha256"])
        if not HEX_SHA256.fullmatch(remote_hash):
            raise ReachabilityError(f"source_receipt_sha256_invalid:{filename}")
        try:
            data = base64.b64decode(
                str(entry["content_base64"]),
                validate=True,
            )
        except ValueError as exc:
            raise ReachabilityError(
                f"source_receipt_base64_invalid:{filename}"
            ) from exc
        if hashlib.sha256(data).hexdigest() != remote_hash:
            raise ReachabilityError(f"source_receipt_hash_mismatch:{filename}")
        path = materialized_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        source_bytes[filename] = data
        file_hashes[filename] = remote_hash
        materialized_paths[filename] = _portable_path(path)

    estimator = json.loads(
        source_bytes["online_estimator_snapshot.json"].decode("utf-8")
    )
    feedback = json.loads(
        source_bytes["fill_feedback_snapshot.json"].decode("utf-8")
    )
    exposures = _csv_rows(
        source_bytes["quote_exposure_intervals.csv"],
        filename="quote_exposure_intervals.csv",
    )
    intensity_rows = _csv_rows(
        source_bytes["online_intensity_fit.csv"],
        filename="online_intensity_fit.csv",
    )
    lifecycle_rows = _csv_rows(
        source_bytes["fill_feedback_lifecycle_matrix.csv"],
        filename="fill_feedback_lifecycle_matrix.csv",
    )
    if any(row.get("side") not in {"buy", "sell"} for row in exposures):
        raise ReachabilityError("exposure_side_invalid")
    if any(_float(row.get("distance_ticks")) is None for row in exposures):
        raise ReachabilityError("exposure_distance_invalid")
    if len(intensity_rows) != 2 or {
        row.get("side") for row in intensity_rows
    } != {"buy", "sell"}:
        raise ReachabilityError("intensity_side_rows_mismatch")
    by_side = {
        side: [row for row in exposures if row.get("side") == side]
        for side in ("buy", "sell")
    }
    distances = {
        side: sorted(
            {
                value
                for value in (
                    _float(row.get("distance_ticks"))
                    for row in by_side[side]
                )
                if value is not None
            }
        )
        for side in ("buy", "sell")
    }
    fit_by_side = {row.get("side"): row for row in intensity_rows}
    if int(estimator.get("quote_exposure_interval_count", -1)) != len(
        exposures
    ):
        raise ReachabilityError("estimator_exposure_count_mismatch")
    for side in ("buy", "sell"):
        if int(fit_by_side.get(side, {}).get("observation_count", -1)) != len(
            by_side[side]
        ):
            raise ReachabilityError(f"intensity_observation_count_mismatch:{side}")

    eligible_rows = [
        row for row in lifecycle_rows
        if _truthy(row.get("included_in_feedback"))
    ]
    eligible_exposure_values = [
        _float(row.get("exposure_seconds")) for row in eligible_rows
    ]
    if any(
        value is None or value < 0 for value in eligible_exposure_values
    ):
        raise ReachabilityError("feedback_eligible_exposure_invalid")
    eligible_exposure = sum(
        value for value in eligible_exposure_values if value is not None
    )
    aggregate = feedback.get("aggregate") or {}
    if int(aggregate.get("lifecycle_count", -1)) != len(lifecycle_rows):
        raise ReachabilityError("feedback_lifecycle_count_mismatch")
    if int(aggregate.get("included_observation_count", -1)) != len(
        eligible_rows
    ):
        raise ReachabilityError("feedback_eligible_count_mismatch")
    aggregate_exposure = _float(aggregate.get("total_exposure_seconds"))
    if aggregate_exposure is None or not math.isclose(
        aggregate_exposure,
        eligible_exposure,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ReachabilityError("feedback_exposure_mismatch")

    dynamic_candidate = estimator.get("dynamic_half_spread_candidate") or {}
    feedback_candidate = feedback.get("candidate") or {}
    return {
        "task_id": package["task_id"],
        "remote_root": package["remote_root"],
        "ssm_command_id": package["command_id"],
        "invocation_receipt_path": _portable_path(receipt_path),
        "invocation_receipt_sha256": _file_hash(receipt_path),
        "file_sha256": file_hashes,
        "materialized_paths": materialized_paths,
        "dynamic": {
            "buy_observation_count": len(by_side["buy"]),
            "sell_observation_count": len(by_side["sell"]),
            "buy_distance_variation_count": len(distances["buy"]),
            "sell_distance_variation_count": len(distances["sell"]),
            "buy_distance_ticks": distances["buy"],
            "sell_distance_ticks": distances["sell"],
            "candidate_status": str(dynamic_candidate.get("status") or ""),
            "candidate_reason": str(dynamic_candidate.get("reason") or ""),
        },
        "fill_feedback": {
            "lifecycle_count": len(lifecycle_rows),
            "eligible_observation_count": len(eligible_rows),
            "eligible_exposure_seconds": eligible_exposure,
            "candidate_status": str(feedback_candidate.get("status") or ""),
            "aggregate_reason": str(aggregate.get("reason") or ""),
        },
    }


def _dynamic_eligible(dynamic: dict[str, Any]) -> bool:
    return (
        int(dynamic["buy_observation_count"])
        >= MIN_DYNAMIC_OBSERVATIONS_PER_SIDE
        and int(dynamic["sell_observation_count"])
        >= MIN_DYNAMIC_OBSERVATIONS_PER_SIDE
        and int(dynamic["buy_distance_variation_count"])
        >= MIN_DYNAMIC_DISTANCE_VARIATIONS_PER_SIDE
        and int(dynamic["sell_distance_variation_count"])
        >= MIN_DYNAMIC_DISTANCE_VARIATIONS_PER_SIDE
    )


def _feedback_eligible(feedback: dict[str, Any]) -> bool:
    return (
        int(feedback["eligible_observation_count"])
        >= MIN_FILL_FEEDBACK_OBSERVATIONS
        and float(feedback["eligible_exposure_seconds"])
        >= MIN_FILL_FEEDBACK_EXPOSURE_SECONDS
    )


def _package_row(package: dict[str, Any], domain: str) -> dict[str, Any]:
    if domain == "dynamic":
        dynamic = package["dynamic"]
        eligible = _dynamic_eligible(dynamic)
        return {
            "scope": package["task_id"],
            "domain": domain,
            "candidate_snapshot_phase": "pre_manager_cycle",
            "evidence_available_phase": "post_manager_cycle",
            "buy_observation_count": dynamic["buy_observation_count"],
            "sell_observation_count": dynamic["sell_observation_count"],
            "buy_distance_variation_count": (
                dynamic["buy_distance_variation_count"]
            ),
            "sell_distance_variation_count": (
                dynamic["sell_distance_variation_count"]
            ),
            "eligible_observation_count": "",
            "eligible_exposure_seconds": "",
            "minimum_requirement": "3 observations and 2 distances per side",
            "reachable_before_submit": False,
            "seed_eligible": eligible,
            "reason": (
                "eligible_source_seed"
                if eligible
                else dynamic["candidate_reason"]
            ),
        }
    feedback = package["fill_feedback"]
    eligible = _feedback_eligible(feedback)
    return {
        "scope": package["task_id"],
        "domain": domain,
        "candidate_snapshot_phase": "pre_manager_cycle",
        "evidence_available_phase": "finalization_after_manager_cycle",
        "buy_observation_count": "",
        "sell_observation_count": "",
        "buy_distance_variation_count": "",
        "sell_distance_variation_count": "",
        "eligible_observation_count": feedback["eligible_observation_count"],
        "eligible_exposure_seconds": feedback["eligible_exposure_seconds"],
        "minimum_requirement": "5 observations and 25 exposure seconds",
        "reachable_before_submit": False,
        "seed_eligible": eligible,
        "reason": (
            "eligible_source_seed"
            if eligible
            else feedback["aggregate_reason"]
        ),
    }


def _combined_rows(packages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dynamic = {
        "buy_observation_count": sum(
            package["dynamic"]["buy_observation_count"]
            for package in packages
        ),
        "sell_observation_count": sum(
            package["dynamic"]["sell_observation_count"]
            for package in packages
        ),
        "buy_distance_variation_count": len(
            {
                value
                for package in packages
                for value in package["dynamic"]["buy_distance_ticks"]
            }
        ),
        "sell_distance_variation_count": len(
            {
                value
                for package in packages
                for value in package["dynamic"]["sell_distance_ticks"]
            }
        ),
    }
    feedback = {
        "eligible_observation_count": sum(
            package["fill_feedback"]["eligible_observation_count"]
            for package in packages
        ),
        "eligible_exposure_seconds": sum(
            package["fill_feedback"]["eligible_exposure_seconds"]
            for package in packages
        ),
    }
    dynamic_eligible = _dynamic_eligible(dynamic)
    feedback_eligible = _feedback_eligible(feedback)
    return [
        {
            "scope": "T047+T052",
            "domain": "dynamic",
            "candidate_snapshot_phase": "pre_manager_cycle",
            "evidence_available_phase": "prior_accepted_packages",
            **dynamic,
            "eligible_observation_count": "",
            "eligible_exposure_seconds": "",
            "minimum_requirement": "3 observations and 2 distances per side",
            "reachable_before_submit": False,
            "seed_eligible": dynamic_eligible,
            "reason": (
                "eligible_source_seed"
                if dynamic_eligible
                else "combined_dynamic_threshold_not_met"
            ),
        },
        {
            "scope": "T047+T052",
            "domain": "fill_feedback",
            "candidate_snapshot_phase": "pre_manager_cycle",
            "evidence_available_phase": "prior_accepted_packages",
            "buy_observation_count": "",
            "sell_observation_count": "",
            "buy_distance_variation_count": "",
            "sell_distance_variation_count": "",
            **feedback,
            "minimum_requirement": "5 observations and 25 exposure seconds",
            "reachable_before_submit": False,
            "seed_eligible": feedback_eligible,
            "reason": (
                "eligible_source_seed"
                if feedback_eligible
                else "combined_fill_feedback_threshold_not_met"
            ),
        },
    ]


def _next_sequence() -> list[dict[str, Any]]:
    return [
        {
            "order": 1,
            "task": "public_multi_distance_dynamic_calibration_and_seed_contract",
            "live_orders": False,
            "exit_gate": (
                "both sides pass intensity fit with >=3 observations and "
                ">=2 distances; seed hash/source pinned"
            ),
        },
        {
            "order": 2,
            "task": "three_window_dynamic_spread_live_evidence",
            "live_orders": True,
            "exit_gate": (
                "dynamic candidate pass and quote changed before submit; "
                "role-known fill economics if fill occurs; >=5 eligible "
                "lifecycles targeted across three windows"
            ),
        },
        {
            "order": 3,
            "task": "fill_feedback_target_ratification",
            "live_orders": False,
            "exit_gate": (
                "accepted lifecycle package supports target proposal and "
                "controller explicitly authorizes target"
            ),
        },
        {
            "order": 4,
            "task": "fill_feedback_active_live_evidence",
            "live_orders": True,
            "exit_gate": (
                "candidate pass and quote changed before submit; role-known "
                "fill/fee/economics evidence when fills occur"
            ),
        },
    ]


def build_artifacts(
    *,
    receipt_index_path: Path = DEFAULT_RECEIPT_INDEX_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    watcher_path: Path = DEFAULT_WATCHER_PATH,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    index = _read_json(receipt_index_path)
    package_index = _validate_receipt_index(index)
    packages = [
        _decode_receipt(package, output_dir=output_dir)
        for package in package_index
    ]
    source_timeline = _source_timeline(watcher_path)
    rows = [
        *(_package_row(package, "dynamic") for package in packages),
        *(_package_row(package, "fill_feedback") for package in packages),
        *_combined_rows(packages),
    ]
    combined_dynamic = next(
        row
        for row in rows
        if row["scope"] == "T047+T052" and row["domain"] == "dynamic"
    )
    combined_feedback = next(
        row
        for row in rows
        if row["scope"] == "T047+T052"
        and row["domain"] == "fill_feedback"
    )
    dynamic_seed_eligible = combined_dynamic["seed_eligible"] is True
    feedback_seed_eligible = combined_feedback["seed_eligible"] is True
    blocking_reasons = ["current_cycle_evidence_arrives_after_quote_decision"]
    if not dynamic_seed_eligible:
        blocking_reasons.append("combined_dynamic_threshold_not_met")
    if not feedback_seed_eligible:
        blocking_reasons.append("combined_fill_feedback_threshold_not_met")
    blocking_reasons.append("fill_feedback_target_not_ratified")
    recommendation = (
        "route_to_source_pinned_dynamic_seed_then_three_window_dynamic_live"
        if dynamic_seed_eligible
        else (
            "route_to_public_multi_distance_dynamic_seed_then_three_window_"
            "dynamic_live"
        )
    )
    current_cycle = {
        "manager_cycle_count": source_timeline["manager_cycle_call_count"],
        "max_submissions": 2,
        "dynamic_snapshot_before_manager_cycle": True,
        "dynamic_exposure_observed_after_manager_cycle": True,
        "fill_feedback_snapshot_before_manager_cycle": True,
        "fill_feedback_lifecycle_built_during_finalization": True,
        "dynamic_activation_reachable_without_seed": False,
        "fill_feedback_activation_reachable_without_seed": False,
    }
    manifest = {
        "task_id": TASK_ID,
        "repair_task_id": REPAIR_TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "receipt_index_path": _portable_path(receipt_index_path),
        "receipt_index_sha256": _file_hash(receipt_index_path),
        "source_snapshot_sha256": _canonical_hash(packages),
        "source_task_ids": [package["task_id"] for package in packages],
        "source_timeline": source_timeline,
        "current_cycle": current_cycle,
        "minimum_dynamic_observations_per_side": (
            MIN_DYNAMIC_OBSERVATIONS_PER_SIDE
        ),
        "minimum_dynamic_distance_variations_per_side": (
            MIN_DYNAMIC_DISTANCE_VARIATIONS_PER_SIDE
        ),
        "minimum_fill_feedback_observations": (
            MIN_FILL_FEEDBACK_OBSERVATIONS
        ),
        "minimum_fill_feedback_exposure_seconds": (
            MIN_FILL_FEEDBACK_EXPOSURE_SECONDS
        ),
        "current_sources_dynamic_seed_eligible": dynamic_seed_eligible,
        "current_sources_fill_feedback_seed_eligible": feedback_seed_eligible,
        "fresh_live_authorization_required_for_sequence_steps": [2, 4],
        "final_recommendation": recommendation,
        "blocking_reasons": blocking_reasons,
        "next_sequence": _next_sequence(),
    }
    boundary = {
        "task_id": TASK_ID,
        "repair_task_id": REPAIR_TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "offline_local_processing_only": True,
        "remote_existing_artifacts_read_only": True,
        "ssm_commands_read_only": True,
        "receipts_and_redacted_source_bytes_materialized": True,
        "no_credentials": True,
        "no_private_account_order_cancel_endpoints": True,
        "no_watcher_or_orchestrator_start": True,
        "no_service_start": True,
        "no_live_orders": True,
        "no_live_authorization_consumed": True,
    }
    _write_json(output_dir / "source_evidence_snapshot.json", {
        "task_id": TASK_ID,
        "repair_task_id": REPAIR_TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "packages": packages,
    })
    _write_csv(output_dir / "reachability_matrix.csv", rows)
    _write_json(output_dir / "live_sequence_contract.json", {
        "task_id": TASK_ID,
        "repair_task_id": REPAIR_TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "sequence": _next_sequence(),
    })
    _write_json(output_dir / "boundary_manifest.json", boundary)
    _write_json(output_dir / "reachability_manifest.json", manifest)
    (output_dir / "recommendation.md").write_text(
        "\n".join(
            [
                "# Adaptive Live Evidence Reachability",
                "",
                f"`{recommendation}`",
                "",
                "- Current single-cycle adaptive activation is unreachable "
                "without a source-pinned seed.",
                f"- Current dynamic seed eligible: `{dynamic_seed_eligible}`.",
                f"- Current fill-feedback seed eligible: "
                f"`{feedback_seed_eligible}`.",
                "- Fresh exact live authorization is required for sequence "
                "steps 2 and 4.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "boundary": boundary,
        "rows": rows,
        "packages": packages,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt-index-path",
        type=Path,
        default=DEFAULT_RECEIPT_INDEX_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--watcher-path", type=Path, default=DEFAULT_WATCHER_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_artifacts(
        receipt_index_path=args.receipt_index_path,
        output_dir=args.output_dir,
        watcher_path=args.watcher_path,
    )
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
