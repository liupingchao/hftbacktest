from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from examples.hyperliquid import (
    cross_exchange_adaptive_live_reachability_preflight as preflight,
)


def _csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _source_files(
    *,
    buy_distances: list[float],
    sell_distances: list[float],
    feedback_exposures: list[float],
    candidate_reason: str = "insufficient_distance_variation",
) -> dict[str, bytes]:
    exposures = [
        {"side": side, "distance_ticks": distance}
        for side, distances in (
            ("buy", buy_distances),
            ("sell", sell_distances),
        )
        for distance in distances
    ]
    lifecycle = [
        {
            "included_in_feedback": "True",
            "exposure_seconds": exposure,
        }
        for exposure in feedback_exposures
    ]
    estimator = {
        "quote_exposure_interval_count": len(exposures),
        "dynamic_half_spread_candidate": {
            "status": "fallback_fixed",
            "reason": candidate_reason,
        },
    }
    feedback = {
        "aggregate": {
            "lifecycle_count": len(lifecycle),
            "included_observation_count": len(lifecycle),
            "total_exposure_seconds": sum(feedback_exposures),
            "reason": (
                "eligible_complete_resting_lifecycle_observations"
                if lifecycle
                else "no_eligible_complete_resting_lifecycle_observations"
            ),
        },
        "candidate": {"status": "unavailable_neutral"},
    }
    intensity = [
        {"side": "buy", "observation_count": len(buy_distances)},
        {"side": "sell", "observation_count": len(sell_distances)},
    ]
    return {
        "online_estimator_snapshot.json": (
            json.dumps(estimator, sort_keys=True) + "\n"
        ).encode(),
        "online_intensity_fit.csv": _csv_bytes(
            intensity,
            ["side", "observation_count"],
        ),
        "quote_exposure_intervals.csv": _csv_bytes(
            exposures,
            ["side", "distance_ticks"],
        ),
        "fill_feedback_snapshot.json": (
            json.dumps(feedback, sort_keys=True) + "\n"
        ).encode(),
        "fill_feedback_lifecycle_matrix.csv": _csv_bytes(
            lifecycle,
            ["included_in_feedback", "exposure_seconds"],
        ),
    }


def _write_watcher(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "def run_event_driven_inline_reprice_live():",
                "    task7_manager_cycle = run_task7_manager_cycle(",
                "        dynamic_spread_candidate=(",
                "        ),",
                "        fill_feedback_candidate=(",
                "        ),",
                "    )",
                "    state.online_estimator.observe_quote_exposure(",
                "    )",
                "    feedback_artifacts = write_fill_feedback_artifacts(",
                "    )",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_package_receipt(
    root: Path,
    *,
    task_id: str,
    command_id: str,
    files: dict[str, bytes],
) -> dict[str, str]:
    remote_root = f"/redacted/{task_id}"
    payload = {
        "schema_version": preflight.RECEIPT_SCHEMA_VERSION,
        "task_id": task_id,
        "remote_root": remote_root,
        "files": {
            filename: {
                "sha256": hashlib.sha256(data).hexdigest(),
                "content_base64": base64.b64encode(data).decode("ascii"),
            }
            for filename, data in files.items()
        },
    }
    receipt_path = root / "receipts" / f"{task_id}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(
            {
                "CommandId": command_id,
                "InstanceId": "i-abc123",
                "Status": "Success",
                "ResponseCode": 0,
                "StandardErrorContent": "",
                "StandardOutputContent": json.dumps(payload, sort_keys=True),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "task_id": task_id,
        "command_id": command_id,
        "remote_root": remote_root,
        "receipt_path": str(receipt_path.relative_to(root)),
    }


def _case(
    root: Path,
    *,
    first: dict[str, list[float]] | None = None,
    second: dict[str, list[float]] | None = None,
) -> tuple[Path, Path, list[Path]]:
    first = first or {
        "buy_distances": [0.5, 0.5, 0.5, 0.5],
        "sell_distances": [11.5, 11.5, 11.5, 11.5],
        "feedback_exposures": [],
    }
    second = second or {
        "buy_distances": [],
        "sell_distances": [28.5, 28.5, 30.5, 31.5],
        "feedback_exposures": [],
    }
    packages = [
        _write_package_receipt(
            root,
            task_id="0721T047",
            command_id="11111111-1111-1111-1111-111111111111",
            files=_source_files(**first),
        ),
        _write_package_receipt(
            root,
            task_id="0722T052",
            command_id="22222222-2222-2222-2222-222222222222",
            files=_source_files(**second),
        ),
    ]
    index_path = root / "source_receipt_index.json"
    index_path.write_text(
        json.dumps(
            {
                "schema_version": preflight.RECEIPT_SCHEMA_VERSION,
                "task_id": preflight.REPAIR_TASK_ID,
                "instance_id": "i-abc123",
                "packages": packages,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    watcher_path = root / "watcher.py"
    _write_watcher(watcher_path)
    return (
        index_path,
        watcher_path,
        [root / package["receipt_path"] for package in packages],
    )


@pytest.fixture
def isolated_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    monkeypatch.setattr(preflight, "PROJECT_ROOT", tmp_path)
    return tmp_path


def _build(root: Path) -> dict[str, Any]:
    index_path, watcher_path, _ = _case(root)
    return preflight.build_artifacts(
        receipt_index_path=index_path,
        output_dir=root / "out",
        watcher_path=watcher_path,
    )


def test_current_receipts_are_derived_as_unreachable(
    isolated_project: Path,
) -> None:
    result = _build(isolated_project)

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
    assert combined_dynamic["sell_observation_count"] == 8
    assert combined_dynamic["buy_distance_variation_count"] == 1
    assert combined_dynamic["sell_distance_variation_count"] == 4
    assert combined_dynamic["seed_eligible"] is False
    assert result["boundary"]["no_live_orders"] is True


def test_eligible_sources_drive_manifest_and_recommendation(
    isolated_project: Path,
) -> None:
    index_path, watcher_path, _ = _case(
        isolated_project,
        first={
            "buy_distances": [1.0, 2.0, 1.0],
            "sell_distances": [1.0, 2.0, 1.0],
            "feedback_exposures": [5.0, 5.0, 5.0],
        },
        second={
            "buy_distances": [2.0, 3.0, 3.0],
            "sell_distances": [2.0, 3.0, 3.0],
            "feedback_exposures": [5.0, 5.0],
        },
    )
    result = preflight.build_artifacts(
        receipt_index_path=index_path,
        output_dir=isolated_project / "out",
        watcher_path=watcher_path,
    )

    manifest = result["manifest"]
    assert manifest["current_sources_dynamic_seed_eligible"] is True
    assert manifest["current_sources_fill_feedback_seed_eligible"] is True
    assert manifest["final_recommendation"] == (
        "route_to_source_pinned_dynamic_seed_then_three_window_dynamic_live"
    )
    assert manifest["blocking_reasons"] == [
        "current_cycle_evidence_arrives_after_quote_decision",
        "fill_feedback_target_not_ratified",
    ]


def _mutate_receipt(
    receipt_path: Path,
    mutate: Callable[[dict[str, Any], dict[str, Any]], None],
) -> None:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload = json.loads(receipt["StandardOutputContent"])
    mutate(receipt, payload)
    receipt["StandardOutputContent"] = json.dumps(payload, sort_keys=True)
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda _receipt, payload: payload["files"][
                "online_estimator_snapshot.json"
            ].__setitem__("sha256", "0" * 64),
            "source_receipt_hash_mismatch",
        ),
        (
            lambda _receipt, payload: payload["files"][
                "online_estimator_snapshot.json"
            ].__setitem__("sha256", "z" * 64),
            "source_receipt_sha256_invalid",
        ),
        (
            lambda _receipt, payload: payload["files"].__setitem__(
                "unexpected.csv",
                payload["files"].pop("online_intensity_fit.csv"),
            ),
            "source_receipt_filenames_mismatch",
        ),
        (
            lambda receipt, _payload: receipt.__setitem__(
                "CommandId",
                "33333333-3333-3333-3333-333333333333",
            ),
            "ssm_invocation_command_id_mismatch",
        ),
    ],
)
def test_receipt_identity_and_hash_drift_fail_closed(
    isolated_project: Path,
    mutation: Callable[[dict[str, Any], dict[str, Any]], None],
    error: str,
) -> None:
    index_path, watcher_path, receipt_paths = _case(isolated_project)
    _mutate_receipt(receipt_paths[0], mutation)

    with pytest.raises(preflight.ReachabilityError, match=error):
        preflight.build_artifacts(
            receipt_index_path=index_path,
            output_dir=isolated_project / "out",
            watcher_path=watcher_path,
        )


def test_source_count_drift_fails_closed(isolated_project: Path) -> None:
    index_path, watcher_path, receipt_paths = _case(isolated_project)

    def mutate(_receipt: dict[str, Any], payload: dict[str, Any]) -> None:
        entry = payload["files"]["online_estimator_snapshot.json"]
        estimator = json.loads(
            base64.b64decode(entry["content_base64"]).decode("utf-8")
        )
        estimator["quote_exposure_interval_count"] = 999
        data = (json.dumps(estimator, sort_keys=True) + "\n").encode()
        entry["content_base64"] = base64.b64encode(data).decode("ascii")
        entry["sha256"] = hashlib.sha256(data).hexdigest()

    _mutate_receipt(receipt_paths[0], mutate)
    with pytest.raises(
        preflight.ReachabilityError,
        match="estimator_exposure_count_mismatch",
    ):
        preflight.build_artifacts(
            receipt_index_path=index_path,
            output_dir=isolated_project / "out",
            watcher_path=watcher_path,
        )


def test_artifacts_are_deterministic(isolated_project: Path) -> None:
    index_path, watcher_path, _ = _case(isolated_project)
    output_dir = isolated_project / "out"
    first = preflight.build_artifacts(
        receipt_index_path=index_path,
        output_dir=output_dir,
        watcher_path=watcher_path,
    )
    first_bytes = {
        path.relative_to(output_dir): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    second = preflight.build_artifacts(
        receipt_index_path=index_path,
        output_dir=output_dir,
        watcher_path=watcher_path,
    )
    second_bytes = {
        path.relative_to(output_dir): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }

    assert first == second
    assert first_bytes == second_bytes
