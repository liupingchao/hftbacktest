from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import basis_positive_filtered_context_viability as viability
import canonical_event_mode_evidence as loader
from test_canonical_directional_momentum_viability import PRICING_FIELDS, _read_csv, _write_csv


def _pricing_rows(sample_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(12):
        basis = [4, 6, 8, 20, 24, 28, 60, 70, -20, -30, -40, -50][index]
        future = [30, -25, 25, 35, 40, -5, 90, 95, -20, -25, -30, -35][index]
        hl_top5 = [-0.2, -0.2, 0.2, 0.2, 0.2, 0.2, 0.25, 0.25, 0.1, 0.1, 0.1, 0.1][index]
        hl_micro = [-0.3, -0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.1, 0.1, 0.1, 0.1][index]
        rows.append(
            {
                "sample_id": sample_id,
                "source_row_index": index,
                "future_row_index": index + 2,
                "effective_future_row_delta": 2,
                "hyperliquid_decision_ts": 1_000_000 + index * 500_000,
                "binance_local_ts": 999_990 + index * 500_000,
                "binance_source_age_ms": 10,
                "joined_row_quality": "primary_usable",
                "horizon_ms": 1000,
                "future_hyperliquid_decision_ts": 1_001_000 + index * 500_000,
                "effective_future_age_ms": 1000,
                "label_row_quality": "primary_label_available",
                "timestamp_policy": "inputs_at_decision_ts_future_labels_at_or_after_target",
                "trade_pressure_policy": "disabled_unverified_side_semantics",
                "basis_contract_caveat": "diagnostic_only",
                "input_binance_top5_imbalance": 0.2,
                "input_binance_microprice_minus_mid_ticks": 0.5,
                "input_binance_mid_move_ticks_from_prev": 0 if index != 2 else "",
                "input_binance_top5_bid_qty": 10,
                "context_hyperliquid_mid_px": 100,
                "context_hyperliquid_spread_ticks": 15,
                "context_hyperliquid_top5_imbalance": hl_top5,
                "context_hyperliquid_microprice_minus_mid_ticks": hl_micro,
                "context_hyperliquid_join_age_ms": 0,
                "context_hyperliquid_join_age_bucket": "fresh_0_50ms",
                "context_hyperliquid_context_quality": "primary_usable",
                "context_basis_mid_ticks": basis,
                "hyperliquid_future_mid_move_ticks": future,
                "hyperliquid_future_microprice_minus_mid_change_ticks": 0,
                "hyperliquid_future_top5_imbalance_change": 0,
                "basis_future_mid_response_ticks": 0,
                "basis_future_microprice_response_ticks": 0,
            }
        )
    for horizon in (100, 250, 5000, 10000):
        for base in rows[:4]:
            extra = dict(base)
            extra["horizon_ms"] = horizon
            extra["effective_future_row_delta"] = max(1, horizon // 500)
            rows.append(extra)
    return rows


def _aggregate(base: Path, *, synthetic: bool = False) -> Path:
    out = base / "aggregate"
    samples = []
    for index in range(5):
        sample_id = f"s{index + 1}"
        pricing_dir = base / sample_id / "pricing"
        mode = "synthetic_fixed_grid" if synthetic else "event"
        status = loader.DIAGNOSTIC_SYNTHETIC_STATUS if synthetic else loader.CANONICAL_EVENT_STATUS
        _write_csv(pricing_dir / "pricing_signal_rows.csv", [] if synthetic else _pricing_rows(sample_id), PRICING_FIELDS)
        samples.append(
            {
                "sample_id": sample_id,
                "decision_mode": mode,
                "canonical_status": status,
                "pricing_signal_dir": str(pricing_dir),
                "source_sample_dir": str(base / sample_id / "source"),
                "source_join_dir": str(base / sample_id / "join"),
                "source_analysis_dir": str(base / sample_id / "analysis"),
                "alignment_run_manifest": str(base / sample_id / "alignment" / "run_manifest.json"),
                "alignment_metrics": str(base / sample_id / "alignment" / "metrics.json"),
                "event_decision_count": 12 if not synthetic else 0,
                "synthetic_decision_count": 0 if not synthetic else 12,
                "run_manifest": str(pricing_dir / "run_manifest.json"),
                "pricing_signal_rows": str(pricing_dir / "pricing_signal_rows.csv"),
            }
        )
    canonical_count = 0 if synthetic else 5
    diagnostic_count = 5 if synthetic else 0
    out.mkdir(parents=True, exist_ok=True)
    (out / "multi_sample_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "binance_led_hyperliquid_multisample_robustness_v1",
                "task_id": "test",
                "sample_count": 5,
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
        [
            {
                "sample_id": sample["sample_id"],
                "pricing_signal_dir": sample["pricing_signal_dir"],
                "source_sample_dir": sample["source_sample_dir"],
                "decision_mode": sample["decision_mode"],
                "canonical_status": sample["canonical_status"],
                "input_rows": 12,
                "primary_rows": 12,
                "pricing_signal_rows": 12,
                "future_join_count": 0,
                "missing_binance_join_count": 0,
                "primary_usable_row_count": 12,
                "horizon_count": 5,
                "independent_future_row_delta_count": 3,
            }
            for sample in samples
        ],
        [
            "sample_id",
            "pricing_signal_dir",
            "source_sample_dir",
            "decision_mode",
            "canonical_status",
            "input_rows",
            "primary_rows",
            "pricing_signal_rows",
            "future_join_count",
            "missing_binance_join_count",
            "primary_usable_row_count",
            "horizon_count",
            "independent_future_row_delta_count",
        ],
    )
    _write_csv(
        out / "feature_horizon_stability_across_samples.csv",
        [
            {
                "feature": "binance_mid_move_ticks_from_prev",
                "horizon_ms": 1000,
                "label": "hyperliquid_future_mid_move_ticks",
                "sample_count": 5,
                "eligible_sample_count": 5,
                "canonical_eligible_sample_count": canonical_count,
                "canonical_independent_future_row_delta_count": 3 if canonical_count else 0,
                "diagnostic_synthetic_sample_count": diagnostic_count,
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
            "diagnostic_synthetic_sample_count",
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
                "row_count": 12,
                "effective_future_age_ms_mean": 1000,
                "effective_future_row_delta_mean": 2,
                "distinct_future_row_delta_count": 2,
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
            "effective_future_age_ms_mean",
            "effective_future_row_delta_mean",
            "distinct_future_row_delta_count",
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
                "row_count": 60 if canonical_count else 0,
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
            "conditioning_status",
        ],
    )
    return out


def _prereq_dirs(base: Path, *, t002_final: str = "tail_filter_hypothesis_validated_for_read_only_research") -> tuple[Path, Path, Path, Path]:
    t002_collection = base / "t002_collection"
    t002_decomp = base / "t002_decomp"
    t001_decomp = base / "t001_decomp"
    t006 = base / "t006"
    for path in (t002_collection, t002_decomp, t001_decomp, t006):
        path.mkdir(parents=True, exist_ok=True)
    (t002_collection / "local_processing_manifest.json").write_text(
        json.dumps(
            {
                "t002_final_recommendation": t002_final,
                "aggregate_canonical_sample_count": 7,
                "decomposition_canonical_sample_count": 7,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (t002_decomp / "basis_positive_wrong_way_manifest.json").write_text(
        json.dumps({"canonical_sample_count": 7, "final_recommendation": "targeted_collection_ready"}) + "\n",
        encoding="utf-8",
    )
    (t001_decomp / "basis_positive_wrong_way_manifest.json").write_text(
        json.dumps({"canonical_sample_count": 3, "final_recommendation": "targeted_collection_ready"}) + "\n",
        encoding="utf-8",
    )
    (t006 / "basis_positive_robustness_manifest.json").write_text(
        json.dumps({"final_recommendation": "needs_more_samples"}) + "\n",
        encoding="utf-8",
    )
    return t002_collection, t002_decomp, t001_decomp, t006


def _build(tmp_path: Path, *, synthetic: bool = False, t002_final: str = "tail_filter_hypothesis_validated_for_read_only_research") -> dict[str, object]:
    t002_collection, t002_decomp, t001_decomp, t006 = _prereq_dirs(tmp_path, t002_final=t002_final)
    return viability.build_filtered_context_viability(
        input_dir=_aggregate(tmp_path, synthetic=synthetic),
        t002_collection_dir=t002_collection,
        t002_decomposition_dir=t002_decomp,
        t001_decomposition_dir=t001_decomp,
        t006_dir=t006,
        output_dir=tmp_path / "out",
    )


def test_writes_required_artifacts_and_labels(tmp_path: Path) -> None:
    result = _build(tmp_path)
    output_dir = tmp_path / "out"
    manifest = json.loads((output_dir / "filtered_context_viability_manifest.json").read_text(encoding="utf-8"))
    labels = _read_csv(output_dir / "research_context_labels.csv")

    assert manifest["schema_version"] == viability.SCHEMA_VERSION
    assert manifest["t002_final_recommendation"] == "tail_filter_hypothesis_validated_for_read_only_research"
    assert manifest["final_recommendation"] in viability.FINAL_RECOMMENDATIONS
    assert manifest["boundary_flags"]["no_order_endpoints"] is True
    assert {row["research_context_label"] for row in labels} == viability.LABELS
    assert all("order side" in row["forbidden_use"] for row in labels)
    assert _read_csv(output_dir / "raw_vs_filtered_basis_positive_summary.csv")
    assert _read_csv(output_dir / "tail_risk_reject_subset_summary.csv")
    assert _read_csv(output_dir / "per_sample_filtered_context_stability.csv")
    assert _read_csv(output_dir / "horizon_filtered_context_stability.csv")
    assert _read_csv(output_dir / "conditioning_filtered_context_summary.csv")
    assert (output_dir / "execution_evidence_gap_register.md").read_text(encoding="utf-8")
    assert (output_dir / "filtered_context_next_step_recommendation.md").read_text(encoding="utf-8")
    assert result["manifest"]["canonical_sample_count"] == 5


def test_clean_context_improves_wrong_way_loss_and_keeps_samples(tmp_path: Path) -> None:
    _build(tmp_path)
    summary = _read_csv(tmp_path / "out" / "raw_vs_filtered_basis_positive_summary.csv")
    clean = next(row for row in summary if row["context_label"] == "basis_positive_clean_context")
    tail = next(row for row in summary if row["context_label"] == "basis_positive_tail_risk_context")

    assert clean["sample_count"] == "5"
    assert float(clean["p95_wrong_way_improvement_vs_raw_ticks"]) > 0
    assert float(tail["wrong_way_rate"]) > float(clean["wrong_way_rate"])


def test_refuses_missing_t002_qa_recommendation(tmp_path: Path) -> None:
    with pytest.raises(viability.FilteredContextInputError):
        _build(tmp_path, t002_final="needs_more_targeted_samples")


def test_refuses_diagnostic_only_synthetic_input(tmp_path: Path) -> None:
    with pytest.raises(loader.EvidenceValidationError):
        _build(tmp_path, synthetic=True)


def test_gap_register_keeps_execution_boundaries(tmp_path: Path) -> None:
    _build(tmp_path)
    text = (tmp_path / "out" / "execution_evidence_gap_register.md").read_text(encoding="utf-8")

    assert "Fill probability is unproven" in text
    assert "Queue position and queue-ahead are unproven" in text
    assert "private/order endpoint" in text
    assert "promotion is authorized" in text
