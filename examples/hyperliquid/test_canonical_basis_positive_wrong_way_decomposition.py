from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import canonical_basis_positive_wrong_way_decomposition as wrong_way
import canonical_event_mode_evidence as loader
from test_canonical_directional_momentum_viability import PRICING_FIELDS, _aggregate, _read_csv, _write_csv


def _t006_dir(base: Path, *, recommendation: str = "needs_more_samples") -> Path:
    out = base / "t006"
    out.mkdir(parents=True, exist_ok=True)
    (out / "basis_positive_robustness_manifest.json").write_text(
        json.dumps(
            {
                "task_id": "0608T006",
                "schema_version": "canonical_basis_positive_robustness_v1",
                "final_recommendation": recommendation,
                "scope_policy": "not_limited_to_regime_011",
                "t005_final_contract_decision": "upgrade_to_context_only_supported",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def _prepare_input(base: Path, *, synthetic: bool = False, weak_control: bool = False) -> Path:
    input_dir = _aggregate(base, synthetic=synthetic)
    if synthetic:
        return input_dir
    manifest = json.loads((input_dir / "multi_sample_manifest.json").read_text(encoding="utf-8"))
    for sample_index, sample in enumerate(manifest["samples"]):
        path = Path(sample["pricing_signal_rows"])
        rows = _read_csv(path)
        updated = []
        for row_index, row in enumerate(rows):
            if row["horizon_ms"] != "1000":
                updated.append(row)
                continue
            if row_index == 0:
                row["context_basis_mid_ticks"] = ""
            elif row_index in {1, 2, 3}:
                row["context_basis_mid_ticks"] = str(5 + 20 * row_index + sample_index)
                if weak_control:
                    row["hyperliquid_future_mid_move_ticks"] = "1"
                else:
                    row["hyperliquid_future_mid_move_ticks"] = "-12" if row_index == 1 and sample_index < 2 else "30"
            else:
                row["context_basis_mid_ticks"] = str(-10 - row_index)
                row["hyperliquid_future_mid_move_ticks"] = "1" if weak_control else "-20"
            row["context_hyperliquid_spread_ticks"] = str(8 + 10 * (row_index % 3))
            row["context_hyperliquid_join_age_ms"] = str(10 + 60 * (row_index % 2))
            row["input_binance_mid_move_ticks_from_prev"] = "" if row_index == 1 else "0"
            row["context_hyperliquid_top5_imbalance"] = str(0.2 if row_index % 2 else -0.2)
            row["context_hyperliquid_microprice_minus_mid_ticks"] = str(0.3 if row_index % 2 else -0.3)
            updated.append(row)
        _write_csv(path, updated, PRICING_FIELDS)
    return input_dir


def _build(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    return wrong_way.build_basis_positive_wrong_way_decomposition(
        input_dir=_prepare_input(tmp_path, **kwargs),
        t006_dir=_t006_dir(tmp_path),
        output_dir=tmp_path / "out",
    )


def test_writes_required_artifacts_and_validates_t006_prerequisite(tmp_path: Path) -> None:
    result = _build(tmp_path)
    output_dir = tmp_path / "out"
    manifest = json.loads((output_dir / "basis_positive_wrong_way_manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == wrong_way.SCHEMA_VERSION
    assert manifest["t006_final_recommendation"] == "needs_more_samples"
    assert manifest["baseline_policy"] == "basis_positive_vs_basis_nonpositive_excluding_missing_or_non_numeric_basis"
    assert manifest["binance_momentum_policy"] == "input_binance_mid_move_ticks_from_prev_only_unavailable_bucket_if_missing"
    assert manifest["final_recommendation"] in wrong_way.FINAL_RECOMMENDATIONS
    assert _read_csv(output_dir / "basis_positive_vs_nonpositive_baseline.csv")
    assert _read_csv(output_dir / "basis_positive_magnitude_bins.csv")
    assert _read_csv(output_dir / "basis_positive_wrong_way_rows.csv")
    assert _read_csv(output_dir / "basis_positive_tail_state_decomposition.csv")
    assert _read_csv(output_dir / "basis_positive_controlled_effects.csv")
    assert _read_csv(output_dir / "basis_positive_tail_filter_feasibility.csv")
    assert (output_dir / "basis_positive_targeted_collection_plan.md").read_text(encoding="utf-8")
    assert (output_dir / "basis_positive_next_step_recommendation.md").read_text(encoding="utf-8")
    assert result["manifest"]["boundary_flags"]["no_order_endpoints"] is True


def test_baseline_excludes_missing_basis_and_extracts_wrong_way_rows(tmp_path: Path) -> None:
    _build(tmp_path)
    baseline = _read_csv(tmp_path / "out" / "basis_positive_vs_nonpositive_baseline.csv")
    wrong_rows = _read_csv(tmp_path / "out" / "basis_positive_wrong_way_rows.csv")

    positive = next(row for row in baseline if row["group_value"] == "basis_positive")
    nonpositive = next(row for row in baseline if row["group_value"] == "basis_nonpositive")
    assert positive["row_count"] == "9"
    assert nonpositive["row_count"] == "6"
    assert len(wrong_rows) == 2
    assert all(float(row["hyperliquid_future_mid_move_ticks"]) < 0 for row in wrong_rows)


def test_magnitude_bins_controlled_effects_and_filter_taxonomy(tmp_path: Path) -> None:
    result = _build(tmp_path)
    magnitude = _read_csv(tmp_path / "out" / "basis_positive_magnitude_bins.csv")
    controlled = _read_csv(tmp_path / "out" / "basis_positive_controlled_effects.csv")
    filters = _read_csv(tmp_path / "out" / "basis_positive_tail_filter_feasibility.csv")

    assert {row["group_value"] for row in magnitude} >= {
        "basis_positive_small",
        "basis_positive_medium",
        "basis_positive_large",
    }
    assert any(row["control_dimension"] == "binance_momentum_bucket" for row in controlled)
    assert any(row["control_dimension"] == "hl_book_state_bucket" for row in controlled)
    assert {row["classification"] for row in filters} <= wrong_way.FILTER_CLASSIFICATIONS
    assert result["manifest"]["controlled_support"]["binance_momentum_bucket"] is True


def test_refuses_diagnostic_only_synthetic_input(tmp_path: Path) -> None:
    with pytest.raises(loader.EvidenceValidationError):
        wrong_way.build_basis_positive_wrong_way_decomposition(
            input_dir=_prepare_input(tmp_path, synthetic=True),
            t006_dir=_t006_dir(tmp_path),
            output_dir=tmp_path / "out",
        )


def test_requires_t006_needs_more_samples_prerequisite(tmp_path: Path) -> None:
    with pytest.raises(wrong_way.BasisPositiveWrongWayInputError):
        wrong_way.build_basis_positive_wrong_way_decomposition(
            input_dir=_prepare_input(tmp_path),
            t006_dir=_t006_dir(tmp_path, recommendation="keep_for_read_only_context_research"),
            output_dir=tmp_path / "out",
        )


def test_rejects_when_controlled_effect_collapses(tmp_path: Path) -> None:
    result = wrong_way.build_basis_positive_wrong_way_decomposition(
        input_dir=_prepare_input(tmp_path, weak_control=True),
        t006_dir=_t006_dir(tmp_path),
        output_dir=tmp_path / "out",
    )

    assert result["manifest"]["final_recommendation"] == "reject_tail_not_filterable"


def test_plan_and_recommendation_keep_forbidden_boundaries(tmp_path: Path) -> None:
    _build(tmp_path)
    plan = (tmp_path / "out" / "basis_positive_targeted_collection_plan.md").read_text(encoding="utf-8")
    recommendation = (tmp_path / "out" / "basis_positive_next_step_recommendation.md").read_text(encoding="utf-8")

    assert "No new data was collected" in plan
    assert "later separately dispatched task with its own QA acceptance" in plan
    assert "No executable trading instruction" in recommendation
    assert "private/order endpoint" in recommendation
    assert "promotion is authorized" in recommendation
