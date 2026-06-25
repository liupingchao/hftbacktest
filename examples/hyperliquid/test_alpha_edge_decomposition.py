from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("alpha_edge_decomposition.py")
SPEC = importlib.util.spec_from_file_location("alpha_edge_decomposition", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    pricing = tmp_path / "pricing"
    canonical = tmp_path / "canonical"
    validation = tmp_path / "validation"
    maker = tmp_path / "maker"
    production = tmp_path / "production"
    output = tmp_path / "output"
    pricing_rows = []
    for horizon in [100, 250, 500, 1000]:
        for index, z in enumerate([-1.0, 1.0]):
            row: dict[str, object] = {
                "horizon_ms": horizon,
                "effective_future_age_ms": 250 if horizon == 100 else horizon + 10,
                "hyperliquid_future_mid_move_ticks": -2.0 if z < 0 else 3.0,
                "hyperliquid_future_microprice_minus_mid_change_ticks": -1.0 if z < 0 else 2.0,
                "context_basis_mid_ticks": -10.0 if z < 0 else 10.0,
                "context_hyperliquid_spread_ticks": 1.0 if z < 0 else 3.0,
                "context_hyperliquid_top5_imbalance": z,
                "context_hyperliquid_microprice_minus_mid_ticks": z * 0.5,
                "context_hyperliquid_join_age_bucket": (
                    "fresh_0_50ms" if z < 0 else "warm_50_250ms"
                ),
            }
            for feature in MODULE.FEATURES:
                row[f"input_{feature}_z"] = z
            pricing_rows.append(row)
    _write_csv(pricing / "pricing_signal_rows.csv", pricing_rows)
    _write_json(
        pricing / "run_manifest.json",
        {"quality": {"source_join_quality": {"future_join_count": 0, "missing_binance_join_count": 0}}},
    )
    canonical_rows = []
    for feature in MODULE.FEATURES:
        for horizon in [100, 250, 500, 1000]:
            canonical_rows.append(
                {
                    "feature": feature,
                    "horizon_ms": horizon,
                    "label": "hyperliquid_future_mid_move_ticks",
                    "canonical_eligible_sample_count": 3,
                    "majority_direction": "positive",
                    "direction_consistency_ratio": 1,
                    "mean_high_minus_low_effect": 5,
                    "mean_abs_corr": 0.2,
                    "stability_verdict": "stable_across_samples",
                }
            )
    _write_csv(canonical / "feature_horizon_stability_across_samples.csv", canonical_rows)
    _write_json(canonical / "multi_sample_manifest.json", {"samples": [{}, {}, {}]})
    _write_json(validation / "canonical_sample_manifest.json", {"samples": [{"sample_id": "a"}, {"sample_id": "b"}, {"sample_id": "c"}]})
    _write_csv(
        maker / "regime_executability_summary.csv",
        [{"public_proxy_row_count": 10, "final_recommendation": "reject_not_maker_executable"}],
    )
    _write_csv(
        maker / "spread_capture_adverse_selection.csv",
        [{"adverse_selection_proxy_ticks": 7}],
    )
    _write_csv(
        production / "current_candidate_audit.csv",
        [{"event_sequence": 1}, {"event_sequence": 2}],
    )
    _write_csv(
        production / "anti_drift_gate_matrix.csv",
        [
            {"status": "block", "reason": "touch_stability_below_minimum"},
            {"status": "pass", "reason": ""},
        ],
    )
    _write_csv(
        production / "fair_mid_source_matrix.csv",
        [
            {
                "event_sequence": 1,
                "source_status": "pass",
                "source_age_ms": 20,
                "basis_mid_ticks": -10,
                "lead_move_ticks": -5,
            },
            {
                "event_sequence": 2,
                "source_status": "block",
                "source_age_ms": 400,
                "basis_mid_ticks": "",
                "lead_move_ticks": "",
            },
        ],
    )
    _write_csv(
        production / "edge_gate_matrix.csv",
        [
            {
                "event_sequence": 1,
                "side": "buy",
                "quote_px": 100,
                "fair_mid_px": 99.5,
                "edge_ticks": -5,
                "edge_gate_reason": "edge_below_required_buffer",
            },
            {
                "event_sequence": 2,
                "side": "buy",
                "quote_px": 100,
                "fair_mid_px": "",
                "edge_ticks": "",
                "edge_gate_reason": "fair_mid_source_stale",
            },
        ],
    )
    _write_json(
        production / "public_shadow_source_manifest.json",
        {
            "fresh_touch_allowed_count": 2,
            "anti_drift_pass_count": 1,
            "anti_drift_block_count": 1,
            "fair_mid_source_pass_count": 1,
            "fair_mid_source_block_count": 1,
            "edge_gate_pass_count": 0,
            "edge_gate_block_count": 2,
        },
    )
    _write_json(production / "boundary_manifest.json", {"no_submit_enforced": True})
    return {
        "pricing": pricing,
        "canonical": canonical,
        "validation": validation,
        "maker": maker,
        "production": production,
        "output": output,
    }


def test_build_artifacts_classifies_insufficient_production_coverage(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = MODULE.build_artifacts(
        pricing_dir=paths["pricing"],
        canonical_dir=paths["canonical"],
        canonical_validation_dir=paths["validation"],
        maker_dir=paths["maker"],
        production_dir=paths["production"],
        output_dir=paths["output"],
    )
    assert result["manifest"]["recommendation"] == "needs_more_public_samples"
    assert result["manifest"]["timestamp_policy"]["future_labels_are_decision_inputs"] is False
    assert result["manifest"]["production_funnel"]["fresh_touch_allowed_count"] == 2
    assert (paths["output"] / "recommendation.md").exists()
    assert len(result["signal_rows"]) == 16
    assert result["manifest"]["effective_horizon_summary"][0]["timing_status"] == "materially_delayed"
    assert result["manifest"]["effective_horizon_summary"][-1]["timing_status"] == "aligned"
    assert result["conditioning_rows"]
    assert (paths["output"] / "venue_state_conditioning.csv").exists()
    production_rows = [row for row in result["lead_move_rows"] if row["evidence_layer"] == "production_public_shadow"]
    assert production_rows[0]["direction_or_side_alignment"] == "opposed"


def test_edge_threshold_sensitivity_uses_only_fresh_numeric_edges() -> None:
    rows = [
        {"edge_ticks": "0.5", "edge_gate_reason": "edge_below_required_buffer"},
        {"edge_ticks": "8", "edge_gate_reason": ""},
        {"edge_ticks": "", "edge_gate_reason": "fair_mid_source_stale"},
    ]
    output = MODULE._edge_sensitivity_rows(rows)
    by_threshold = {int(row["threshold_ticks"]): row for row in output}
    assert by_threshold[0]["pass_count"] == 2
    assert by_threshold[7]["pass_count"] == 1
    assert by_threshold[7]["fresh_edge_row_count"] == 2


def test_outputs_are_deterministic_for_same_inputs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    kwargs = {
        "pricing_dir": paths["pricing"],
        "canonical_dir": paths["canonical"],
        "canonical_validation_dir": paths["validation"],
        "maker_dir": paths["maker"],
        "production_dir": paths["production"],
        "output_dir": paths["output"],
    }
    MODULE.build_artifacts(**kwargs)
    first = {
        path.name: path.read_bytes()
        for path in sorted(paths["output"].iterdir())
        if path.is_file()
    }
    MODULE.build_artifacts(**kwargs)
    second = {
        path.name: path.read_bytes()
        for path in sorted(paths["output"].iterdir())
        if path.is_file()
    }
    assert first == second


def test_effective_horizon_reports_delay_and_distribution(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = MODULE.build_artifacts(
        pricing_dir=paths["pricing"],
        canonical_dir=paths["canonical"],
        canonical_validation_dir=paths["validation"],
        maker_dir=paths["maker"],
        production_dir=paths["production"],
        output_dir=paths["output"],
    )
    row = next(
        row
        for row in result["signal_rows"]
        if row["feature"] == MODULE.FEATURES[0] and row["horizon_ms"] == 100
    )
    assert row["effective_age_count"] == 2
    assert row["effective_age_min_ms"] == "250"
    assert row["effective_age_mean_ms"] == "250"
    assert row["effective_age_max_ms"] == "250"
    assert row["effective_age_offset_mean_ms"] == "150"
    assert row["timing_status"] == "materially_delayed"


def test_venue_state_conditioning_is_fail_closed_on_small_buckets(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = MODULE.build_artifacts(
        pricing_dir=paths["pricing"],
        canonical_dir=paths["canonical"],
        canonical_validation_dir=paths["validation"],
        maker_dir=paths["maker"],
        production_dir=paths["production"],
        output_dir=paths["output"],
    )
    basis_rows = [
        row
        for row in result["conditioning_rows"]
        if row["condition_name"] == "basis_mid_ticks" and row["horizon_ms"] == 100
    ]
    assert {row["bucket"] for row in basis_rows} == {"low", "high"}
    assert all(row["coverage_status"] == "insufficient_coverage" for row in basis_rows)
    cause = next(
        row for row in result["root_cause_rows"] if row["root_cause"] == "basis_conditioning"
    )
    assert cause["assessment"] == "insufficient_coverage"


def test_venue_state_conditioning_reports_material_covered_buckets() -> None:
    rows = []
    for index in range(80):
        low = index < 40
        rows.append(
            {
                "horizon_ms": "1000",
                "hyperliquid_future_mid_move_ticks": "-2" if low else "3",
                "hyperliquid_future_microprice_minus_mid_change_ticks": "-1" if low else "2",
                "context_basis_mid_ticks": "-10" if low else "10",
                "context_hyperliquid_spread_ticks": "1" if low else "3",
                "context_hyperliquid_top5_imbalance": "-0.5" if low else "0.5",
                "context_hyperliquid_microprice_minus_mid_ticks": "-0.25" if low else "0.25",
                "context_hyperliquid_join_age_bucket": "fresh_0_50ms" if low else "warm_50_250ms",
            }
        )
    output = MODULE._venue_state_conditioning_rows(rows)
    basis_rows = [
        row
        for row in output
        if row["condition_name"] == "basis_mid_ticks" and row["horizon_ms"] == 1000
    ]
    assert {row["bucket"] for row in basis_rows} == {"low", "high"}
    assert all(row["coverage_status"] == "sufficient" for row in basis_rows)
    assessment, evidence_count, finding = MODULE._conditioning_assessment(
        output, {"basis_mid_ticks"}
    )
    assert assessment == "material"
    assert evidence_count == 80
    assert "5" in finding
