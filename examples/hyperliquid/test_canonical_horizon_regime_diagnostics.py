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
import canonical_horizon_regime_diagnostics as diagnostics


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


def _sample(sample_id: str, *, decision_mode: str = "event", canonical_status: str = loader.CANONICAL_EVENT_STATUS) -> dict[str, object]:
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
        "event_decision_count": 100 if decision_mode == "event" else 0,
        "synthetic_decision_count": 0 if decision_mode == "event" else 100,
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
        "input_rows": 100,
        "primary_rows": 100,
        "excluded_rows": 0,
        "pricing_signal_rows": 600,
        "future_join_count": 0,
        "missing_binance_join_count": 0,
        "primary_usable_row_count": 100,
        "effective_future_age_ms_min": 100,
        "effective_future_age_ms_mean": 1000,
        "effective_future_age_ms_max": 5000,
        "horizon_count": 2,
        "independent_future_row_delta_count": 4,
        "horizon_future_row_delta_groups": "100:1,2|1000:1,2,3,4",
        "aliased_horizon_signature_count": 0,
        "pricing_signal_recommendation": "keep_for_read_only_research",
        "single_public_sample_caveat": "True",
        "quality_status": sample["canonical_status"],
    }


def _make_aggregate(
    base: Path,
    *,
    samples: list[dict[str, object]] | None = None,
    aliasing_rows: list[dict[str, object]] | None = None,
    venue_rows: list[dict[str, object]] | None = None,
) -> Path:
    out = base / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    samples = samples or [_sample("s1"), _sample("s2"), _sample("s3")]
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
        "schema_version": "test",
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
                "horizon_ms": horizon,
                "label": "hyperliquid_future_mid_move_ticks",
                "sample_count": len(samples),
                "eligible_sample_count": len(samples),
                "canonical_eligible_sample_count": canonical_count,
                "canonical_independent_future_row_delta_count": 4 if horizon >= 1000 else 2,
                "majority_direction": "positive" if canonical_count else "insufficient",
                "direction_consistency_ratio": 1 if canonical_count else 0,
                "positive_sample_count": canonical_count,
                "negative_sample_count": 0,
                "zero_sample_count": 0,
                "diagnostic_synthetic_sample_count": diagnostic_count,
                "total_row_count": 100 * canonical_count,
                "mean_high_minus_low_effect": 1 if canonical_count else "",
                "std_high_minus_low_effect": 0,
                "mean_abs_corr": 0.2 if canonical_count else "",
                "sample_effects": "",
                "stability_verdict": "stable_across_samples" if canonical_count else loader.DIAGNOSTIC_SYNTHETIC_STATUS,
            }
            for horizon in [100, 1000]
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
    if aliasing_rows is None:
        aliasing_rows = []
        for sample in samples:
            for horizon, distinct_count, row_delta_mean in [(100, 2, 1.1), (1000, 4, 3.5)]:
                aliasing_rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "decision_mode": sample["decision_mode"],
                        "canonical_status": sample["canonical_status"],
                        "horizon_ms": horizon,
                        "row_count": 100,
                        "effective_future_age_ms_min": horizon,
                        "effective_future_age_ms_mean": horizon + 100,
                        "effective_future_age_ms_max": horizon + 200,
                        "effective_future_row_delta_min": 1,
                        "effective_future_row_delta_mean": row_delta_mean,
                        "effective_future_row_delta_max": distinct_count,
                        "distinct_future_row_delta_count": distinct_count,
                        "distinct_effective_age_count": 50,
                        "mean_age_alias_group": horizon,
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
    if venue_rows is None:
        venue_rows = [
            {
                "horizon_ms": 100,
                "hyperliquid_context_quality": "primary_usable",
                "hyperliquid_join_age_bucket": "fresh_0_50ms",
                "hyperliquid_spread_bucket": "spread_0_10_ticks",
                "sample_count": 3,
                "row_count": 300,
                "mean_future_mid_move_ticks": 1,
                "sample_means": "s1:1|s2:1.5|s3:2",
                "conditioning_status": "multi_sample_regime",
            },
            {
                "horizon_ms": 1000,
                "hyperliquid_context_quality": "primary_usable",
                "hyperliquid_join_age_bucket": "fresh_0_50ms",
                "hyperliquid_spread_bucket": "spread_0_10_ticks",
                "sample_count": 3,
                "row_count": 300,
                "mean_future_mid_move_ticks": 1,
                "sample_means": "s1:1|s2:1.5|s3:2",
                "conditioning_status": "multi_sample_regime",
            },
        ]
    _write_csv(
        out / "venue_state_conditioning_across_samples.csv",
        venue_rows,
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


def test_horizon_alias_classification_marks_short_horizons_watch_only(tmp_path: Path) -> None:
    input_dir = _make_aggregate(tmp_path)

    result = diagnostics.build_horizon_regime_diagnostics(input_dir=input_dir, output_dir=tmp_path / "out")

    by_horizon = {int(row["horizon_ms"]): row for row in result["horizon_rows"]}
    assert by_horizon[100]["support_status"] == "watch_needs_more_samples"
    assert by_horizon[100]["independence_status"] == "weak_short_horizon_watch"
    assert by_horizon[1000]["support_status"] == "diagnostic_supported"
    assert by_horizon[1000]["independence_status"] == "independent_enough_for_formal_diagnostic"


def test_minimum_support_classification_rejects_thin_regime_bucket(tmp_path: Path) -> None:
    input_dir = _make_aggregate(
        tmp_path,
        venue_rows=[
            {
                "horizon_ms": 1000,
                "hyperliquid_context_quality": "primary_usable",
                "hyperliquid_join_age_bucket": "fresh_0_50ms",
                "hyperliquid_spread_bucket": "spread_gt_20_ticks",
                "sample_count": 1,
                "row_count": 20,
                "mean_future_mid_move_ticks": 5,
                "sample_means": "s1:5",
                "conditioning_status": "single_sample_regime",
            }
        ],
    )

    result = diagnostics.build_horizon_regime_diagnostics(input_dir=input_dir, output_dir=tmp_path / "out")

    assert result["regime_rows"][0]["support_status"] == "reject_insufficient_support"


def test_concentration_penalty_rejects_single_sample_dominated_regime(tmp_path: Path) -> None:
    input_dir = _make_aggregate(
        tmp_path,
        venue_rows=[
            {
                "horizon_ms": 1000,
                "hyperliquid_context_quality": "primary_usable",
                "hyperliquid_join_age_bucket": "fresh_0_50ms",
                "hyperliquid_spread_bucket": "spread_10_20_ticks",
                "sample_count": 3,
                "row_count": 300,
                "mean_future_mid_move_ticks": 34,
                "sample_means": "s1:100|s2:1|s3:1",
                "conditioning_status": "multi_sample_regime",
            }
        ],
    )

    result = diagnostics.build_horizon_regime_diagnostics(input_dir=input_dir, output_dir=tmp_path / "out")

    assert result["regime_rows"][0]["support_status"] == "reject_aliased_or_concentrated"
    assert result["regime_rows"][0]["caveat"] == "sample_effect_concentration_too_high"


def test_refuses_diagnostic_only_synthetic_input(tmp_path: Path) -> None:
    input_dir = _make_aggregate(
        tmp_path,
        samples=[
            _sample(
                "s1_synth",
                decision_mode="synthetic_fixed_grid",
                canonical_status=loader.DIAGNOSTIC_SYNTHETIC_STATUS,
            )
        ],
        venue_rows=[],
    )

    with pytest.raises(diagnostics.DiagnosticsInputError, match="no canonical event-mode samples"):
        diagnostics.build_horizon_regime_diagnostics(input_dir=input_dir, output_dir=tmp_path / "out")


def test_report_artifact_generation(tmp_path: Path) -> None:
    input_dir = _make_aggregate(tmp_path)
    output_dir = tmp_path / "out"

    result = diagnostics.build_horizon_regime_diagnostics(input_dir=input_dir, output_dir=output_dir)

    manifest_path = output_dir / "horizon_regime_diagnostics_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == diagnostics.SCHEMA_VERSION
    assert manifest["boundary_flags"]["no_strategy_implementation"] is True
    assert result["watch_rows"]
    assert _read_csv(output_dir / "horizon_independence_diagnostics.csv")
    assert _read_csv(output_dir / "regime_conditioning_diagnostics.csv")
    assert _read_csv(output_dir / "regime_watch_list.csv")
    assert "no final regime selection" in (
        output_dir / "horizon_regime_diagnostics_report.md"
    ).read_text(encoding="utf-8").lower()
