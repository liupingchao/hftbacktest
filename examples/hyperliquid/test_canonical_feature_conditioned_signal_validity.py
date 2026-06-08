from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import canonical_directional_momentum_viability as momentum
import canonical_event_mode_evidence as loader
import canonical_feature_conditioned_signal_validity as validity
from test_canonical_directional_momentum_viability import (
    _aggregate,
    _candidate_dir,
    _maker_dir,
    _prereq_dirs,
    _read_csv,
)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _contract_dir(base: Path) -> Path:
    out = base / "contract"
    rows = [
        {
            "feature": "binance_top5_imbalance",
            "venue": "Binance",
            "role": "lead_pricing_pressure",
            "decision": "allow",
            "status": "primary_allowlist",
        },
        {
            "feature": "binance_microprice_minus_mid_ticks",
            "venue": "Binance",
            "role": "lead_microprice_dislocation",
            "decision": "allow",
            "status": "primary_allowlist",
        },
        {
            "feature": "binance_mid_move_ticks_from_prev",
            "venue": "Binance",
            "role": "lead_short_horizon_momentum",
            "decision": "allow",
            "status": "primary_allowlist",
        },
        {
            "feature": "binance_top5_bid_qty",
            "venue": "Binance",
            "role": "lead_asymmetric_liquidity_pressure",
            "decision": "allow",
            "status": "primary_allowlist",
        },
        {
            "feature": "hyperliquid_top5_imbalance",
            "venue": "Hyperliquid",
            "role": "venue_state",
            "decision": "allow",
            "status": "context_only",
        },
        {
            "feature": "hyperliquid_microprice_minus_mid_ticks",
            "venue": "Hyperliquid",
            "role": "venue_state",
            "decision": "allow",
            "status": "context_only",
        },
        {
            "feature": "basis_mid_dislocation",
            "venue": "Joint",
            "role": "basis_context",
            "decision": "diagnostic_only",
            "status": "diagnostic_context",
        },
    ]
    _write_csv(
        out / "feature_decision_table.csv",
        rows,
        ["feature", "venue", "role", "decision", "status"],
    )
    return out


def _t003_dir(base: Path, *, final: str = "reject_directional_edge_unstable") -> Path:
    out = base / "t003"
    out.mkdir(parents=True, exist_ok=True)
    (out / "directional_momentum_manifest.json").write_text(
        json.dumps(
            {
                "task_id": "0608T003",
                "schema_version": momentum.SCHEMA_VERSION,
                "assessed_regime_id": momentum.TARGET_REGIME_ID,
                "final_recommendation": final,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        out / "directional_candidate_watch_reject.csv",
        [{"regime_id": momentum.TARGET_REGIME_ID, "final_recommendation": final}],
        ["regime_id", "final_recommendation"],
    )
    return out


def _make_redundant_features(input_dir: Path) -> None:
    manifest = json.loads((input_dir / "multi_sample_manifest.json").read_text(encoding="utf-8"))
    for sample in manifest["samples"]:
        path = Path(sample["pricing_signal_rows"])
        rows = _read_csv(path)
        for index, row in enumerate(rows):
            value = (index % 6) - 2
            row["input_binance_top5_imbalance"] = value
            row["input_binance_microprice_minus_mid_ticks"] = value * 2
        _write_csv(path, rows, list(rows[0]))


def _build(
    tmp_path: Path,
    *,
    synthetic: bool = False,
    maker_final: str = "reject_not_maker_executable",
    t003_final: str = "reject_directional_edge_unstable",
) -> dict[str, object]:
    input_dir = _aggregate(tmp_path, synthetic=synthetic)
    if not synthetic:
        _make_redundant_features(input_dir)
    candidate_dir = _candidate_dir(tmp_path)
    maker_dir = _maker_dir(tmp_path, final=maker_final)
    signal_dir, horizon_dir = _prereq_dirs(tmp_path)
    return validity.build_feature_conditioned_signal_validity(
        input_dir=input_dir,
        candidate_dir=candidate_dir,
        maker_executability_dir=maker_dir,
        directional_momentum_dir=_t003_dir(tmp_path, final=t003_final),
        signal_ranking_dir=signal_dir,
        horizon_regime_dir=horizon_dir,
        data_contract_dir=_contract_dir(tmp_path),
        output_dir=tmp_path / "out",
    )


def test_accepts_canonical_input_and_writes_required_artifacts(tmp_path: Path) -> None:
    result = _build(tmp_path)
    out = tmp_path / "out"
    manifest = json.loads((out / "feature_validity_manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == validity.SCHEMA_VERSION
    assert manifest["assessed_regime_id"] == momentum.TARGET_REGIME_ID
    assert manifest["row_level_input_policy"] == "multi_sample_manifest.samples[].pricing_signal_rows"
    assert manifest["maker_prerequisite_final_recommendation"] == "reject_not_maker_executable"
    assert manifest["directional_momentum_prerequisite_final_recommendation"] == "reject_directional_edge_unstable"
    assert manifest["final_recommendation"] in validity.NEXT_STEP_RECOMMENDATIONS
    assert _read_csv(out / "feature_pattern_validity_summary.csv")
    assert _read_csv(out / "per_sample_pattern_stability.csv")
    assert _read_csv(out / "feature_redundancy_collinearity.csv")
    assert _read_csv(out / "decision_visibility_caveat_audit.csv")
    assert _read_csv(out / "cost_tail_validity.csv")
    assert result["manifest"]["boundary_flags"]["no_shadow_decision_generation"] is True


def test_refuses_diagnostic_only_synthetic_input(tmp_path: Path) -> None:
    with pytest.raises(loader.EvidenceValidationError):
        _build(tmp_path, synthetic=True)


def test_requires_t002_and_t003_prerequisites(tmp_path: Path) -> None:
    with pytest.raises(momentum.DirectionalMomentumInputError):
        _build(tmp_path / "maker_case", maker_final="candidate_for_maker_case_library")

    with pytest.raises(validity.FeatureValidityInputError):
        _build(tmp_path / "t003_case", t003_final="candidate_for_directional_case_library")


def test_decision_visibility_caveat_classification(tmp_path: Path) -> None:
    _build(tmp_path)
    visibility = _read_csv(tmp_path / "out" / "decision_visibility_caveat_audit.csv")
    by_feature = {row["feature"]: row for row in visibility}

    assert by_feature["input_binance_top5_imbalance"]["visibility_classification"] == "decision_time_visible"
    assert (
        by_feature["context_basis_mid_ticks"]["visibility_classification"]
        == "contract_caveated_diagnostic_context_only"
    )


def test_redundancy_cost_tail_and_final_taxonomy(tmp_path: Path) -> None:
    result = _build(tmp_path)
    summary = _read_csv(tmp_path / "out" / "feature_pattern_validity_summary.csv")
    redundancy = _read_csv(tmp_path / "out" / "feature_redundancy_collinearity.csv")
    cost_tail = _read_csv(tmp_path / "out" / "cost_tail_validity.csv")

    assert any(row["redundancy_classification"] == "high_redundancy" for row in redundancy)
    assert any(row["signal_validity"] == "invalid_redundant_or_leakage_risk" for row in summary)
    assert all(row["cost_adjusted_viability"] in momentum.COST_CLASSIFICATIONS for row in cost_tail)
    assert all(row["tail_risk_classification"] in momentum.TAIL_CLASSIFICATIONS for row in cost_tail)
    assert result["manifest"]["final_recommendation"] in validity.NEXT_STEP_RECOMMENDATIONS
