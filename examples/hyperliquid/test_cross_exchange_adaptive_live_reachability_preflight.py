from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.hyperliquid import (
    cross_exchange_adaptive_live_reachability_preflight as preflight,
)


def _source() -> dict[str, object]:
    def package(
        task_id: str,
        *,
        buy_count: int,
        sell_count: int,
        buy_distances: list[float],
        sell_distances: list[float],
    ) -> dict[str, object]:
        return {
            "task_id": task_id,
            "remote_root": f"/redacted/{task_id}",
            "ssm_command_id": "command-id",
            "file_sha256": {
                f"file_{index}": str(index) * 64
                for index in range(1, 6)
            },
            "dynamic": {
                "buy_observation_count": buy_count,
                "sell_observation_count": sell_count,
                "buy_distance_variation_count": len(set(buy_distances)),
                "sell_distance_variation_count": len(set(sell_distances)),
                "buy_distance_ticks": buy_distances,
                "sell_distance_ticks": sell_distances,
                "candidate_status": "fallback_fixed",
                "candidate_reason": "insufficient_distance_variation",
            },
            "fill_feedback": {
                "lifecycle_count": 2,
                "eligible_observation_count": 0,
                "eligible_exposure_seconds": 0.0,
                "candidate_status": "unavailable_neutral",
                "aggregate_reason": (
                    "no_eligible_complete_resting_lifecycle_observations"
                ),
            },
        }

    return {
        "task_id": preflight.TASK_ID,
        "schema_version": preflight.SCHEMA_VERSION,
        "packages": [
            package(
                "0721T047",
                buy_count=4,
                sell_count=4,
                buy_distances=[0.5],
                sell_distances=[11.5],
            ),
            package(
                "0722T052",
                buy_count=0,
                sell_count=4,
                buy_distances=[],
                sell_distances=[28.5, 30.5, 31.5],
            ),
        ],
    }


def test_current_profile_and_prior_sources_are_unreachable(tmp_path: Path) -> None:
    input_path = tmp_path / "source.json"
    input_path.write_text(json.dumps(_source()), encoding="utf-8")

    result = preflight.build_artifacts(
        input_path=input_path,
        output_dir=tmp_path / "out",
    )

    manifest = result["manifest"]
    assert manifest["current_cycle"][
        "dynamic_activation_reachable_without_seed"
    ] is False
    assert manifest["current_cycle"][
        "fill_feedback_activation_reachable_without_seed"
    ] is False
    assert manifest["current_sources_dynamic_seed_eligible"] is False
    assert manifest["current_sources_fill_feedback_seed_eligible"] is False
    assert manifest["final_recommendation"] == (
        "route_to_public_multi_distance_dynamic_seed_then_three_window_"
        "dynamic_live"
    )
    combined_dynamic = next(
        row
        for row in result["rows"]
        if row["scope"] == "T047+T052" and row["domain"] == "dynamic"
    )
    assert combined_dynamic["buy_observation_count"] == 4
    assert combined_dynamic["buy_distance_variation_count"] == 1
    assert combined_dynamic["seed_eligible"] is False
    assert result["boundary"]["no_live_orders"] is True


def test_source_snapshot_fails_closed_on_hash_or_field_drift(
    tmp_path: Path,
) -> None:
    source = _source()
    source["packages"][0]["unexpected"] = True
    input_path = tmp_path / "bad_fields.json"
    input_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(
        preflight.ReachabilityError,
        match="source_snapshot_package_fields_mismatch",
    ):
        preflight.build_artifacts(
            input_path=input_path,
            output_dir=tmp_path / "out1",
        )

    source = _source()
    source["packages"][0]["file_sha256"]["file_1"] = "bad"
    input_path = tmp_path / "bad_hash.json"
    input_path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(
        preflight.ReachabilityError,
        match="source_snapshot_hashes_invalid",
    ):
        preflight.build_artifacts(
            input_path=input_path,
            output_dir=tmp_path / "out2",
        )


def test_artifacts_are_deterministic(tmp_path: Path) -> None:
    input_path = tmp_path / "source.json"
    input_path.write_text(json.dumps(_source()), encoding="utf-8")
    first = preflight.build_artifacts(
        input_path=input_path,
        output_dir=tmp_path / "out1",
    )
    second = preflight.build_artifacts(
        input_path=input_path,
        output_dir=tmp_path / "out2",
    )

    assert first == second
    assert (
        (tmp_path / "out1" / "reachability_manifest.json").read_bytes()
        == (tmp_path / "out2" / "reachability_manifest.json").read_bytes()
    )

