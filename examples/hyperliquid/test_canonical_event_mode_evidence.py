from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import canonical_event_mode_evidence as loader


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _sample(sample_id: str, *, decision_mode: str, canonical_status: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "decision_mode": decision_mode,
        "canonical_status": canonical_status,
        "pricing_signal_dir": f"/tmp/{sample_id}/pricing",
        "source_sample_dir": f"/tmp/{sample_id}/source",
        "source_join_dir": f"/tmp/{sample_id}/join",
        "source_analysis_dir": f"/tmp/{sample_id}/analysis",
        "alignment_run_manifest": f"/tmp/{sample_id}/alignment/run_manifest.json",
        "alignment_metrics": f"/tmp/{sample_id}/alignment/metrics.json",
        "event_decision_count": 10 if decision_mode == "event" else 0,
        "synthetic_decision_count": 0 if decision_mode == "event" else 10,
        "run_manifest": f"/tmp/{sample_id}/pricing/run_manifest.json",
        "pricing_signal_rows": f"/tmp/{sample_id}/pricing/pricing_signal_rows.csv",
    }


def _quality_row(sample: dict[str, object]) -> dict[str, object]:
    return {
        "sample_id": sample["sample_id"],
        "pricing_signal_dir": sample["pricing_signal_dir"],
        "source_join_dir": sample["source_join_dir"],
        "source_analysis_dir": sample["source_analysis_dir"],
        "source_sample_dir": sample["source_sample_dir"],
        "decision_mode": sample["decision_mode"],
        "canonical_status": sample["canonical_status"],
        "alignment_run_manifest": sample["alignment_run_manifest"],
        "alignment_metrics": sample["alignment_metrics"],
        "event_decision_count": sample["event_decision_count"],
        "synthetic_decision_count": sample["synthetic_decision_count"],
        "input_rows": 10,
        "primary_rows": 10,
        "excluded_rows": 0,
        "pricing_signal_rows": 20,
        "future_join_count": 0,
        "missing_binance_join_count": 0,
        "primary_usable_row_count": 10,
        "effective_future_age_ms_min": 100,
        "effective_future_age_ms_mean": 500,
        "effective_future_age_ms_max": 1000,
        "horizon_count": 2,
        "independent_future_row_delta_count": 2,
        "horizon_future_row_delta_groups": "100:1|250:2",
        "aliased_horizon_signature_count": 0,
        "pricing_signal_recommendation": "keep_for_read_only_research",
        "single_public_sample_caveat": "True",
        "quality_status": sample["canonical_status"],
    }


def _make_aggregate(base: Path, samples: list[dict[str, object]]) -> Path:
    out = base / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    canonical_count = sum(
        1
        for sample in samples
        if sample["decision_mode"] == loader.CANONICAL_DECISION_MODE
        and sample["canonical_status"] == loader.CANONICAL_EVENT_STATUS
    )
    diagnostic_count = sum(
        1 for sample in samples if sample["canonical_status"] == loader.DIAGNOSTIC_SYNTHETIC_STATUS
    )
    manifest = {
        "schema_version": "binance_led_hyperliquid_multisample_robustness_v1",
        "task_id": "test",
        "sample_count": len(samples),
        "canonical_sample_count": canonical_count,
        "diagnostic_synthetic_sample_count": diagnostic_count,
        "samples": samples,
        "boundary_flags": loader.BOUNDARY_FLAGS,
    }
    (out / "multi_sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_csv(
        out / "sample_quality_matrix.csv",
        [_quality_row(sample) for sample in samples],
        [
            "sample_id",
            "pricing_signal_dir",
            "source_join_dir",
            "source_analysis_dir",
            "source_sample_dir",
            "decision_mode",
            "canonical_status",
            "alignment_run_manifest",
            "alignment_metrics",
            "event_decision_count",
            "synthetic_decision_count",
            "input_rows",
            "primary_rows",
            "excluded_rows",
            "pricing_signal_rows",
            "future_join_count",
            "missing_binance_join_count",
            "primary_usable_row_count",
            "effective_future_age_ms_min",
            "effective_future_age_ms_mean",
            "effective_future_age_ms_max",
            "horizon_count",
            "independent_future_row_delta_count",
            "horizon_future_row_delta_groups",
            "aliased_horizon_signature_count",
            "pricing_signal_recommendation",
            "single_public_sample_caveat",
            "quality_status",
        ],
    )
    _write_csv(
        out / "feature_horizon_stability_across_samples.csv",
        [
            {
                "feature": "binance_top5_imbalance",
                "horizon_ms": 100,
                "label": "hyperliquid_future_mid_move_ticks",
                "sample_count": len(samples),
                "eligible_sample_count": len(samples),
                "canonical_eligible_sample_count": canonical_count,
                "canonical_independent_future_row_delta_count": 1 if canonical_count else 0,
                "majority_direction": "positive" if canonical_count else "insufficient",
                "direction_consistency_ratio": 1 if canonical_count else 0,
                "positive_sample_count": canonical_count,
                "negative_sample_count": 0,
                "zero_sample_count": 0,
                "diagnostic_synthetic_sample_count": diagnostic_count,
                "total_row_count": 20 * canonical_count,
                "mean_high_minus_low_effect": 1 if canonical_count else "",
                "std_high_minus_low_effect": 0,
                "mean_abs_corr": 1 if canonical_count else "",
                "sample_effects": "",
                "stability_verdict": "stable_across_samples" if canonical_count else loader.DIAGNOSTIC_SYNTHETIC_STATUS,
            }
        ],
        [
            "feature",
            "horizon_ms",
            "label",
            "sample_count",
            "eligible_sample_count",
            "canonical_eligible_sample_count",
            "canonical_independent_future_row_delta_count",
            "majority_direction",
            "direction_consistency_ratio",
            "positive_sample_count",
            "negative_sample_count",
            "zero_sample_count",
            "diagnostic_synthetic_sample_count",
            "total_row_count",
            "mean_high_minus_low_effect",
            "std_high_minus_low_effect",
            "mean_abs_corr",
            "sample_effects",
            "stability_verdict",
        ],
    )
    aliasing_rows: list[dict[str, object]] = []
    for sample in samples:
        aliasing_rows.append(
            {
                "sample_id": sample["sample_id"],
                "decision_mode": sample["decision_mode"],
                "canonical_status": sample["canonical_status"],
                "horizon_ms": 100,
                "row_count": 10,
                "effective_future_age_ms_min": 100,
                "effective_future_age_ms_mean": 125,
                "effective_future_age_ms_max": 150,
                "effective_future_row_delta_min": 1,
                "effective_future_row_delta_mean": 1,
                "effective_future_row_delta_max": 1,
                "distinct_future_row_delta_count": 1,
                "distinct_effective_age_count": 10,
                "mean_age_alias_group": 100,
                "aliasing_status": "distinct_mean_effective_age",
            }
        )
    _write_csv(
        out / "effective_horizon_aliasing_by_sample.csv",
        aliasing_rows,
        [
            "sample_id",
            "decision_mode",
            "canonical_status",
            "horizon_ms",
            "row_count",
            "effective_future_age_ms_min",
            "effective_future_age_ms_mean",
            "effective_future_age_ms_max",
            "effective_future_row_delta_min",
            "effective_future_row_delta_mean",
            "effective_future_row_delta_max",
            "distinct_future_row_delta_count",
            "distinct_effective_age_count",
            "mean_age_alias_group",
            "aliasing_status",
        ],
    )
    _write_csv(
        out / "venue_state_conditioning_across_samples.csv",
        [
            {
                "horizon_ms": 100,
                "hyperliquid_context_quality": "primary_usable",
                "hyperliquid_join_age_bucket": "fresh_0_50ms",
                "hyperliquid_spread_bucket": "spread_0_10_ticks",
                "sample_count": max(1, canonical_count),
                "row_count": 10 * max(1, canonical_count),
                "mean_future_mid_move_ticks": 1,
                "sample_means": "",
                "conditioning_status": "multi_sample_regime" if canonical_count > 1 else "single_sample_regime",
            }
        ],
        [
            "horizon_ms",
            "hyperliquid_context_quality",
            "hyperliquid_join_age_bucket",
            "hyperliquid_spread_bucket",
            "sample_count",
            "row_count",
            "mean_future_mid_move_ticks",
            "sample_means",
            "conditioning_status",
        ],
    )
    return out


def test_accepts_event_mode_samples_and_writes_canonical_outputs(tmp_path: Path) -> None:
    samples = [
        _sample("sample_a_event", decision_mode="event", canonical_status=loader.CANONICAL_EVENT_STATUS),
        _sample("sample_b_event", decision_mode="event", canonical_status=loader.CANONICAL_EVENT_STATUS),
    ]
    input_dir = _make_aggregate(tmp_path, samples)
    output_dir = tmp_path / "out"

    result = loader.build_canonical_event_mode_evidence_artifacts(input_dir=input_dir, output_dir=output_dir)

    assert result["canonical_sample_count"] == 2
    assert result["diagnostic_rejection_count"] == 0
    manifest = json.loads((output_dir / "canonical_sample_manifest.json").read_text(encoding="utf-8"))
    assert manifest["canonical_sample_count"] == 2
    assert {sample["sample_id"] for sample in manifest["samples"]} == {"sample_a_event", "sample_b_event"}
    quality = _read_csv(output_dir / "canonical_sample_quality_summary.csv")
    assert {row["validation_status"] for row in quality} == {"accepted_canonical_event_mode"}
    assert _read_csv(output_dir / "diagnostic_rejection_report.csv") == []
    assert (output_dir / "canonical_evidence_validation_report.md").exists()


def test_rejects_synthetic_fixed_grid_from_canonical_outputs(tmp_path: Path) -> None:
    input_dir = _make_aggregate(
        tmp_path,
        [
            _sample(
                "sample_synth",
                decision_mode="synthetic",
                canonical_status=loader.DIAGNOSTIC_SYNTHETIC_STATUS,
            )
        ],
    )
    output_dir = tmp_path / "out"

    result = loader.build_canonical_event_mode_evidence_artifacts(input_dir=input_dir, output_dir=output_dir)

    assert result["canonical_sample_count"] == 0
    assert result["diagnostic_rejection_count"] == 1
    manifest = json.loads((output_dir / "canonical_sample_manifest.json").read_text(encoding="utf-8"))
    assert manifest["samples"] == []
    rejections = _read_csv(output_dir / "diagnostic_rejection_report.csv")
    assert rejections[0]["sample_id"] == "sample_synth"
    assert rejections[0]["rejection_reason"] == "diagnostic_only_synthetic_decision_grid_excluded_from_canonical_evidence"


def test_mixed_manifest_keeps_event_and_excludes_diagnostic(tmp_path: Path) -> None:
    input_dir = _make_aggregate(
        tmp_path,
        [
            _sample("sample_event", decision_mode="event", canonical_status=loader.CANONICAL_EVENT_STATUS),
            _sample(
                "sample_synth",
                decision_mode="synthetic",
                canonical_status=loader.DIAGNOSTIC_SYNTHETIC_STATUS,
            ),
        ],
    )
    output_dir = tmp_path / "out"

    result = loader.build_canonical_event_mode_evidence_artifacts(input_dir=input_dir, output_dir=output_dir)

    assert result["canonical_sample_count"] == 1
    assert result["diagnostic_rejection_count"] == 1
    manifest = json.loads((output_dir / "canonical_sample_manifest.json").read_text(encoding="utf-8"))
    assert [sample["sample_id"] for sample in manifest["samples"]] == ["sample_event"]


def test_missing_required_file_fails_clearly(tmp_path: Path) -> None:
    input_dir = _make_aggregate(
        tmp_path,
        [_sample("sample_event", decision_mode="event", canonical_status=loader.CANONICAL_EVENT_STATUS)],
    )
    (input_dir / "venue_state_conditioning_across_samples.csv").unlink()

    with pytest.raises(FileNotFoundError, match="venue_state_conditioning_across_samples"):
        loader.load_canonical_event_mode_evidence(input_dir=input_dir)


def test_missing_required_column_fails_clearly(tmp_path: Path) -> None:
    input_dir = _make_aggregate(
        tmp_path,
        [_sample("sample_event", decision_mode="event", canonical_status=loader.CANONICAL_EVENT_STATUS)],
    )
    rows = _read_csv(input_dir / "sample_quality_matrix.csv")
    rows_without_status = [{key: value for key, value in row.items() if key != "canonical_status"} for row in rows]
    _write_csv(input_dir / "sample_quality_matrix.csv", rows_without_status, list(rows_without_status[0]))

    with pytest.raises(loader.EvidenceValidationError, match="canonical_status"):
        loader.load_canonical_event_mode_evidence(input_dir=input_dir)
