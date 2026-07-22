#!/usr/bin/env python3
"""Build the no-order adaptive-pricing reachability preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0722T064"
SCHEMA_VERSION = "cross_exchange_adaptive_live_reachability_v1"
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "adaptive_live_evidence_reachability_0722T064"
    / "source_evidence_snapshot.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_PATH.parent
DEFAULT_WATCHER_PATH = (
    PROJECT_ROOT
    / "examples"
    / "hyperliquid"
    / "hyperliquid_tiny_live_m2_public_watcher.py"
)
MIN_DYNAMIC_OBSERVATIONS_PER_SIDE = 3
MIN_DYNAMIC_DISTANCE_VARIATIONS_PER_SIDE = 2
MIN_FILL_FEEDBACK_OBSERVATIONS = 5
MIN_FILL_FEEDBACK_EXPOSURE_SECONDS = 25.0


class ReachabilityError(ValueError):
    """Raised when the task-local source snapshot is invalid."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReachabilityError("source_snapshot_must_be_object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
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
        "watcher_path": str(watcher_path.resolve().relative_to(PROJECT_ROOT)),
        "watcher_sha256": hashlib.sha256(
            watcher_path.read_bytes()
        ).hexdigest(),
        "manager_cycle_call_count": section.count(
            tokens["manager_cycle_call"]
        ),
        "line_numbers": lines,
        "dynamic_candidate_is_manager_call_argument": True,
        "dynamic_evidence_observed_after_manager_returns": True,
        "fill_feedback_candidate_is_manager_call_argument": True,
        "fill_feedback_lifecycle_built_after_loop_finalization": True,
    }


def _validate_source(source: dict[str, Any]) -> list[dict[str, Any]]:
    if source.get("schema_version") != SCHEMA_VERSION:
        raise ReachabilityError("source_snapshot_schema_mismatch")
    if source.get("task_id") != TASK_ID:
        raise ReachabilityError("source_snapshot_task_id_mismatch")
    packages = source.get("packages")
    if not isinstance(packages, list) or len(packages) != 2:
        raise ReachabilityError("source_snapshot_packages_mismatch")
    validated: list[dict[str, Any]] = []
    for package in packages:
        if not isinstance(package, dict):
            raise ReachabilityError("source_snapshot_package_invalid")
        required = {
            "task_id",
            "remote_root",
            "ssm_command_id",
            "file_sha256",
            "dynamic",
            "fill_feedback",
        }
        if set(package) != required:
            raise ReachabilityError("source_snapshot_package_fields_mismatch")
        hashes = package["file_sha256"]
        if (
            not isinstance(hashes, dict)
            or len(hashes) != 5
            or any(
                not isinstance(value, str) or len(value) != 64
                for value in hashes.values()
            )
        ):
            raise ReachabilityError("source_snapshot_hashes_invalid")
        dynamic = package["dynamic"]
        feedback = package["fill_feedback"]
        if not isinstance(dynamic, dict) or not isinstance(feedback, dict):
            raise ReachabilityError("source_snapshot_evidence_invalid")
        validated.append(package)
    return validated


def _package_row(package: dict[str, Any], domain: str) -> dict[str, Any]:
    task_id = str(package["task_id"])
    if domain == "dynamic":
        dynamic = package["dynamic"]
        buy_observations = int(dynamic["buy_observation_count"])
        sell_observations = int(dynamic["sell_observation_count"])
        buy_variations = int(dynamic["buy_distance_variation_count"])
        sell_variations = int(dynamic["sell_distance_variation_count"])
        eligible = (
            buy_observations >= MIN_DYNAMIC_OBSERVATIONS_PER_SIDE
            and sell_observations >= MIN_DYNAMIC_OBSERVATIONS_PER_SIDE
            and buy_variations >= MIN_DYNAMIC_DISTANCE_VARIATIONS_PER_SIDE
            and sell_variations >= MIN_DYNAMIC_DISTANCE_VARIATIONS_PER_SIDE
        )
        return {
            "scope": task_id,
            "domain": domain,
            "candidate_snapshot_phase": "pre_manager_cycle",
            "evidence_available_phase": "post_manager_cycle",
            "buy_observation_count": buy_observations,
            "sell_observation_count": sell_observations,
            "buy_distance_variation_count": buy_variations,
            "sell_distance_variation_count": sell_variations,
            "eligible_observation_count": "",
            "eligible_exposure_seconds": "",
            "minimum_requirement": "3 observations and 2 distances per side",
            "reachable_before_submit": False,
            "seed_eligible": eligible,
            "reason": (
                "eligible_source_seed"
                if eligible
                else str(dynamic["candidate_reason"])
            ),
        }
    feedback = package["fill_feedback"]
    eligible_observations = int(feedback["eligible_observation_count"])
    eligible_exposure = float(feedback["eligible_exposure_seconds"])
    eligible = (
        eligible_observations >= MIN_FILL_FEEDBACK_OBSERVATIONS
        and eligible_exposure >= MIN_FILL_FEEDBACK_EXPOSURE_SECONDS
    )
    return {
        "scope": task_id,
        "domain": domain,
        "candidate_snapshot_phase": "pre_manager_cycle",
        "evidence_available_phase": "finalization_after_manager_cycle",
        "buy_observation_count": "",
        "sell_observation_count": "",
        "buy_distance_variation_count": "",
        "sell_distance_variation_count": "",
        "eligible_observation_count": eligible_observations,
        "eligible_exposure_seconds": eligible_exposure,
        "minimum_requirement": "5 observations and 25 exposure seconds",
        "reachable_before_submit": False,
        "seed_eligible": eligible,
        "reason": (
            "eligible_source_seed"
            if eligible
            else str(feedback["aggregate_reason"])
        ),
    }


def _combined_rows(packages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buy_observations = sum(
        int(package["dynamic"]["buy_observation_count"])
        for package in packages
    )
    sell_observations = sum(
        int(package["dynamic"]["sell_observation_count"])
        for package in packages
    )
    buy_distances = {
        float(value)
        for package in packages
        for value in package["dynamic"]["buy_distance_ticks"]
    }
    sell_distances = {
        float(value)
        for package in packages
        for value in package["dynamic"]["sell_distance_ticks"]
    }
    feedback_observations = sum(
        int(package["fill_feedback"]["eligible_observation_count"])
        for package in packages
    )
    feedback_exposure = sum(
        float(package["fill_feedback"]["eligible_exposure_seconds"])
        for package in packages
    )
    return [
        {
            "scope": "T047+T052",
            "domain": "dynamic",
            "candidate_snapshot_phase": "pre_manager_cycle",
            "evidence_available_phase": "prior_accepted_packages",
            "buy_observation_count": buy_observations,
            "sell_observation_count": sell_observations,
            "buy_distance_variation_count": len(buy_distances),
            "sell_distance_variation_count": len(sell_distances),
            "eligible_observation_count": "",
            "eligible_exposure_seconds": "",
            "minimum_requirement": "3 observations and 2 distances per side",
            "reachable_before_submit": False,
            "seed_eligible": (
                buy_observations >= MIN_DYNAMIC_OBSERVATIONS_PER_SIDE
                and sell_observations >= MIN_DYNAMIC_OBSERVATIONS_PER_SIDE
                and len(buy_distances)
                >= MIN_DYNAMIC_DISTANCE_VARIATIONS_PER_SIDE
                and len(sell_distances)
                >= MIN_DYNAMIC_DISTANCE_VARIATIONS_PER_SIDE
            ),
            "reason": "buy_side_insufficient_distance_variation",
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
            "eligible_observation_count": feedback_observations,
            "eligible_exposure_seconds": feedback_exposure,
            "minimum_requirement": "5 observations and 25 exposure seconds",
            "reachable_before_submit": False,
            "seed_eligible": (
                feedback_observations >= MIN_FILL_FEEDBACK_OBSERVATIONS
                and feedback_exposure >= MIN_FILL_FEEDBACK_EXPOSURE_SECONDS
            ),
            "reason": "no_eligible_complete_resting_lifecycle_observations",
        },
    ]


def build_artifacts(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    watcher_path: Path = DEFAULT_WATCHER_PATH,
) -> dict[str, Any]:
    source = _read_json(input_path)
    packages = _validate_source(source)
    source_timeline = _source_timeline(watcher_path)
    rows = [
        *(_package_row(package, "dynamic") for package in packages),
        *(_package_row(package, "fill_feedback") for package in packages),
        *_combined_rows(packages),
    ]
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
    next_sequence = [
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
    recommendation = (
        "route_to_public_multi_distance_dynamic_seed_then_three_window_"
        "dynamic_live"
    )
    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "source_snapshot_sha256": _canonical_hash(source),
        "source_timeline": source_timeline,
        "source_task_ids": [package["task_id"] for package in packages],
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
        "current_sources_dynamic_seed_eligible": False,
        "current_sources_fill_feedback_seed_eligible": False,
        "fresh_live_authorization_required_for_sequence_steps": [2, 4],
        "final_recommendation": recommendation,
        "blocking_reasons": [
            "current_cycle_evidence_arrives_after_quote_decision",
            "combined_dynamic_buy_side_has_one_distance",
            "combined_fill_feedback_has_zero_eligible_lifecycles",
            "fill_feedback_target_not_ratified",
        ],
        "next_sequence": next_sequence,
    }
    boundary = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "offline_local_processing_only": True,
        "remote_existing_artifacts_read_only": True,
        "ssm_commands_read_only": True,
        "no_credentials": True,
        "no_private_account_order_cancel_endpoints": True,
        "no_watcher_or_orchestrator_start": True,
        "no_service_start": True,
        "no_live_orders": True,
        "no_live_authorization_consumed": True,
    }
    output_dir = output_dir.resolve()
    _write_csv(output_dir / "reachability_matrix.csv", rows)
    _write_json(output_dir / "live_sequence_contract.json", {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "sequence": next_sequence,
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
                "- T047 and T052 are not eligible dynamic or fill-feedback "
                "seeds.",
                "- Build public multi-distance intensity evidence before the "
                "next real order.",
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
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--watcher-path", type=Path, default=DEFAULT_WATCHER_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_artifacts(
        input_path=args.input_path,
        output_dir=args.output_dir,
        watcher_path=args.watcher_path,
    )
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
