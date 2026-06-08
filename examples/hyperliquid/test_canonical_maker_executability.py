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
import canonical_maker_executability as maker


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


def _pricing_rows(sample_id: str, *, count: int = 6, spread: float = 15.0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        future_move = 1.0 if index % 2 == 0 else -1.0
        rows.append(
            {
                "sample_id": sample_id,
                "source_row_index": index,
                "future_row_index": index + 3,
                "effective_future_row_delta": 3,
                "hyperliquid_decision_ts": 1_000_000 + index * 500_000,
                "binance_local_ts": 999_990 + index * 500_000,
                "binance_source_age_ms": 10 + index,
                "joined_row_quality": "primary_usable",
                "horizon_ms": 1000,
                "future_hyperliquid_decision_ts": 1_001_000 + index * 500_000,
                "effective_future_age_ms": 1000,
                "label_row_quality": "primary_label_available",
                "timestamp_policy": "inputs_at_decision_ts_future_labels_at_or_after_target",
                "trade_pressure_policy": "disabled_unverified_side_semantics",
                "basis_contract_caveat": "diagnostic_only",
                "input_binance_top5_imbalance": 0.2,
                "input_binance_top5_imbalance_z": 0.1,
                "input_binance_microprice_minus_mid_ticks": 0.5,
                "input_binance_microprice_minus_mid_ticks_z": 0.1,
                "input_binance_mid_move_ticks_from_prev": 2.0 if index % 3 else -1.0,
                "input_binance_mid_move_ticks_from_prev_z": 0.2,
                "input_binance_top5_bid_qty": 10,
                "input_binance_top5_bid_qty_z": 0.1,
                "context_hyperliquid_mid_px": 100,
                "context_hyperliquid_spread_ticks": spread,
                "context_hyperliquid_top5_imbalance": 0.1,
                "context_hyperliquid_microprice_minus_mid_ticks": 0.2,
                "context_hyperliquid_join_age_ms": 0,
                "context_hyperliquid_join_age_bucket": "fresh_0_50ms",
                "context_hyperliquid_context_quality": "primary_usable",
                "context_basis_mid_ticks": 0,
                "hyperliquid_future_mid_move_ticks": future_move,
                "hyperliquid_future_microprice_minus_mid_change_ticks": 0,
                "hyperliquid_future_top5_imbalance_change": 0,
                "basis_future_mid_response_ticks": 0,
                "basis_future_microprice_response_ticks": 0,
            }
        )
    return rows


def _sample(base: Path, sample_id: str, *, synthetic: bool = False) -> dict[str, object]:
    pricing_dir = base / sample_id / "pricing"
    status = loader.DIAGNOSTIC_SYNTHETIC_STATUS if synthetic else loader.CANONICAL_EVENT_STATUS
    mode = "synthetic_fixed_grid" if synthetic else "event"
    rows = [] if synthetic else _pricing_rows(sample_id)
    _write_csv(
        pricing_dir / "pricing_signal_rows.csv",
        rows,
        [
            "sample_id",
            "source_row_index",
            "future_row_index",
            "effective_future_row_delta",
            "hyperliquid_decision_ts",
            "binance_local_ts",
            "binance_source_age_ms",
            "joined_row_quality",
            "horizon_ms",
            "future_hyperliquid_decision_ts",
            "effective_future_age_ms",
            "label_row_quality",
            "timestamp_policy",
            "trade_pressure_policy",
            "basis_contract_caveat",
            "input_binance_top5_imbalance",
            "input_binance_top5_imbalance_z",
            "input_binance_microprice_minus_mid_ticks",
            "input_binance_microprice_minus_mid_ticks_z",
            "input_binance_mid_move_ticks_from_prev",
            "input_binance_mid_move_ticks_from_prev_z",
            "input_binance_top5_bid_qty",
            "input_binance_top5_bid_qty_z",
            "context_hyperliquid_mid_px",
            "context_hyperliquid_spread_ticks",
            "context_hyperliquid_top5_imbalance",
            "context_hyperliquid_microprice_minus_mid_ticks",
            "context_hyperliquid_join_age_ms",
            "context_hyperliquid_join_age_bucket",
            "context_hyperliquid_context_quality",
            "context_basis_mid_ticks",
            "hyperliquid_future_mid_move_ticks",
            "hyperliquid_future_microprice_minus_mid_change_ticks",
            "hyperliquid_future_top5_imbalance_change",
            "basis_future_mid_response_ticks",
            "basis_future_microprice_response_ticks",
        ],
    )
    return {
        "sample_id": sample_id,
        "decision_mode": mode,
        "canonical_status": status,
        "pricing_signal_dir": str(pricing_dir),
        "source_sample_dir": str(base / sample_id / "source"),
        "source_join_dir": str(base / sample_id / "join"),
        "source_analysis_dir": str(base / sample_id / "analysis"),
        "alignment_run_manifest": str(base / sample_id / "alignment" / "run_manifest.json"),
        "alignment_metrics": str(base / sample_id / "alignment" / "metrics.json"),
        "event_decision_count": 6 if not synthetic else 0,
        "synthetic_decision_count": 0 if not synthetic else 6,
        "run_manifest": str(pricing_dir / "run_manifest.json"),
        "pricing_signal_rows": str(pricing_dir / "pricing_signal_rows.csv"),
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
        "input_rows": 60,
        "primary_rows": 60,
        "excluded_rows": 0,
        "pricing_signal_rows": 6,
        "future_join_count": 0,
        "missing_binance_join_count": 0,
        "primary_usable_row_count": 60,
        "effective_future_age_ms_min": 1000,
        "effective_future_age_ms_mean": 1000,
        "effective_future_age_ms_max": 1000,
        "horizon_count": 1,
        "independent_future_row_delta_count": 3,
        "horizon_future_row_delta_groups": "1000:1,2,3",
        "aliased_horizon_signature_count": 0,
        "pricing_signal_recommendation": "keep_for_read_only_research",
        "single_public_sample_caveat": "True",
        "quality_status": sample["canonical_status"],
    }


def _aggregate(base: Path, *, synthetic: bool = False) -> Path:
    out = base / "aggregate"
    samples = [_sample(base, f"s{index}", synthetic=synthetic) for index in range(1, 4)]
    canonical_count = 0 if synthetic else 3
    diagnostic_count = 3 if synthetic else 0
    out.mkdir(parents=True, exist_ok=True)
    (out / "multi_sample_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "binance_led_hyperliquid_multisample_robustness_v1",
                "task_id": "test",
                "sample_count": 3,
                "canonical_sample_count": canonical_count,
                "diagnostic_synthetic_sample_count": diagnostic_count,
                "samples": samples,
                "boundary_flags": loader.BOUNDARY_FLAGS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
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
                "feature": maker.TARGET_ANCHOR_FEATURE,
                "horizon_ms": 1000,
                "label": "hyperliquid_future_mid_move_ticks",
                "sample_count": 3,
                "eligible_sample_count": 3,
                "canonical_eligible_sample_count": canonical_count,
                "canonical_independent_future_row_delta_count": 3 if canonical_count else 0,
                "majority_direction": "positive" if canonical_count else "insufficient",
                "direction_consistency_ratio": 1 if canonical_count else 0,
                "positive_sample_count": canonical_count,
                "negative_sample_count": 0,
                "zero_sample_count": 0,
                "diagnostic_synthetic_sample_count": diagnostic_count,
                "total_row_count": 18 if canonical_count else 0,
                "mean_high_minus_low_effect": 1 if canonical_count else "",
                "std_high_minus_low_effect": 0,
                "mean_abs_corr": 0.1 if canonical_count else "",
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
    _write_csv(
        out / "effective_horizon_aliasing_by_sample.csv",
        [
            {
                "sample_id": sample["sample_id"],
                "decision_mode": sample["decision_mode"],
                "canonical_status": sample["canonical_status"],
                "horizon_ms": 1000,
                "row_count": 6,
                "effective_future_age_ms_min": 1000,
                "effective_future_age_ms_mean": 1000,
                "effective_future_age_ms_max": 1000,
                "effective_future_row_delta_min": 3,
                "effective_future_row_delta_mean": 3,
                "effective_future_row_delta_max": 3,
                "distinct_future_row_delta_count": 3,
                "distinct_effective_age_count": 6,
                "mean_age_alias_group": 1000,
                "aliasing_status": "distinct_mean_effective_age",
            }
            for sample in samples
        ],
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
                "horizon_ms": 1000,
                "hyperliquid_context_quality": "primary_usable",
                "hyperliquid_join_age_bucket": "fresh_0_50ms",
                "hyperliquid_spread_bucket": "spread_10_20_ticks",
                "sample_count": canonical_count,
                "row_count": 18 if canonical_count else 0,
                "mean_future_mid_move_ticks": 0,
                "sample_means": "s1:0|s2:0|s3:0" if canonical_count else "",
                "conditioning_status": "multi_sample_regime" if canonical_count else "diagnostic_only",
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


def _candidate_dir(base: Path, *, extra_candidate: bool = False, row_count: int = 18) -> Path:
    out = base / "candidate"
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "regime_id": maker.TARGET_CANDIDATE_ID,
            "classification": "candidate_for_milestone3_executability",
            "primary_anchor_feature": maker.TARGET_ANCHOR_FEATURE,
            "horizon_ms": 1000,
            "hyperliquid_context_quality": "primary_usable",
            "hyperliquid_join_age_bucket": "fresh_0_50ms",
            "hyperliquid_spread_bucket": "spread_10_20_ticks",
            "row_count": row_count,
            "sample_count": 3,
        }
    ]
    if extra_candidate:
        rows.append(
            {
                "regime_id": "regime_999_1000_spread_gt_20_ticks",
                "classification": "watch_needs_more_samples",
                "primary_anchor_feature": maker.TARGET_ANCHOR_FEATURE,
                "horizon_ms": 1000,
                "hyperliquid_context_quality": "primary_usable",
                "hyperliquid_join_age_bucket": "fresh_0_50ms",
                "hyperliquid_spread_bucket": "spread_gt_20_ticks",
                "row_count": 3,
                "sample_count": 1,
            }
        )
    _write_csv(
        out / "candidate_regime_definitions.csv",
        rows,
        [
            "regime_id",
            "classification",
            "primary_anchor_feature",
            "horizon_ms",
            "hyperliquid_context_quality",
            "hyperliquid_join_age_bucket",
            "hyperliquid_spread_bucket",
            "row_count",
            "sample_count",
        ],
    )
    (out / "canonical_regime_synthesis_manifest.json").write_text(
        json.dumps(
            {
                "task_id": "0604T009",
                "schema_version": "canonical_regime_synthesis_v1",
                "canonical_sample_count": 3,
                "diagnostic_rejection_count": 0,
                "candidate_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def _prereq_dirs(base: Path, *, diagnostic_rejection_count: int = 0) -> tuple[Path, Path]:
    signal = base / "signal"
    horizon = base / "horizon"
    signal.mkdir()
    horizon.mkdir()
    (signal / "signal_quality_ranking_manifest.json").write_text(
        json.dumps(
            {
                "task_id": "0604T006",
                "schema_version": "canonical_signal_quality_ranking_v1",
                "canonical_sample_count": 3,
                "diagnostic_rejection_count": diagnostic_rejection_count,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (horizon / "horizon_regime_diagnostics_manifest.json").write_text(
        json.dumps(
            {
                "task_id": "0604T007",
                "schema_version": "canonical_horizon_regime_diagnostics_v1",
                "canonical_sample_count": 3,
                "diagnostic_rejection_count": diagnostic_rejection_count,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return signal, horizon


def _build(tmp_path: Path, *, candidate_rows: int = 18) -> dict[str, object]:
    input_dir = _aggregate(tmp_path)
    candidate_dir = _candidate_dir(tmp_path, extra_candidate=True, row_count=candidate_rows)
    signal_dir, horizon_dir = _prereq_dirs(tmp_path)
    output_dir = tmp_path / "out"
    return maker.build_canonical_maker_executability(
        input_dir=input_dir,
        candidate_dir=candidate_dir,
        signal_ranking_dir=signal_dir,
        horizon_regime_dir=horizon_dir,
        output_dir=output_dir,
    )


def test_accepts_canonical_input_and_writes_required_outputs(tmp_path: Path) -> None:
    result = _build(tmp_path)
    output_dir = tmp_path / "out"
    manifest = json.loads((output_dir / "maker_executability_manifest.json").read_text(encoding="utf-8"))
    summary = _read_csv(output_dir / "regime_executability_summary.csv")

    assert manifest["schema_version"] == maker.SCHEMA_VERSION
    assert manifest["assessed_candidate_id"] == maker.TARGET_CANDIDATE_ID
    assert manifest["ignored_candidate_count"] == 1
    assert manifest["boundary_flags"]["no_order_endpoints"] is True
    assert summary[0]["proxy_scope"] == "read_only_public_data_proxy_not_private_execution_proof"
    assert _read_csv(output_dir / "fill_opportunity_proxy.csv")
    assert _read_csv(output_dir / "spread_capture_adverse_selection.csv")
    assert _read_csv(output_dir / "quote_churn_post_only_latency.csv")
    assert _read_csv(output_dir / "inventory_exposure_what_if.csv")


def test_refuses_diagnostic_only_synthetic_input(tmp_path: Path) -> None:
    input_dir = _aggregate(tmp_path, synthetic=True)
    candidate_dir = _candidate_dir(tmp_path)
    signal_dir, horizon_dir = _prereq_dirs(tmp_path)

    with pytest.raises(loader.EvidenceValidationError):
        maker.build_canonical_maker_executability(
            input_dir=input_dir,
            candidate_dir=candidate_dir,
            signal_ranking_dir=signal_dir,
            horizon_regime_dir=horizon_dir,
            output_dir=tmp_path / "out",
        )


def test_final_taxonomy_generation_uses_watch_when_execution_proof_missing(tmp_path: Path) -> None:
    result = _build(tmp_path)
    recommendation = result["manifest"]["final_recommendation"]

    assert recommendation in maker.FINAL_RECOMMENDATIONS
    assert recommendation == "watch_needs_execution_evidence"


def test_candidate_row_mismatch_creates_more_public_samples_watch_path(tmp_path: Path) -> None:
    result = _build(tmp_path, candidate_rows=99)
    summary = result["summary_rows"][0]

    assert summary["final_recommendation"] == "watch_needs_more_public_samples"


def test_rejects_prerequisite_manifest_with_diagnostic_rejections(tmp_path: Path) -> None:
    input_dir = _aggregate(tmp_path)
    candidate_dir = _candidate_dir(tmp_path)
    signal_dir, horizon_dir = _prereq_dirs(tmp_path, diagnostic_rejection_count=1)

    with pytest.raises(maker.MakerExecutabilityInputError):
        maker.build_canonical_maker_executability(
            input_dir=input_dir,
            candidate_dir=candidate_dir,
            signal_ranking_dir=signal_dir,
            horizon_regime_dir=horizon_dir,
            output_dir=tmp_path / "out",
        )
