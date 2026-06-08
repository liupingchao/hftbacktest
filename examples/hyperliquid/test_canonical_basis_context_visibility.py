from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import canonical_basis_context_visibility as basis
import canonical_directional_momentum_viability as momentum
import canonical_event_mode_evidence as loader
from test_canonical_directional_momentum_viability import _aggregate, _read_csv


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _contract_dir(base: Path) -> Path:
    out = base / "contract"
    _write_csv(
        out / "feature_decision_table.csv",
        [
            {
                "feature": "basis_mid_dislocation",
                "venue": "Joint",
                "role": "basis_context",
                "decision": "diagnostic_only",
                "status": "diagnostic_context",
            }
        ],
        ["feature", "venue", "role", "decision", "status"],
    )
    return out


def _feature_validity_dir(base: Path, *, valid: bool = True) -> Path:
    out = base / "feature_validity"
    out.mkdir(parents=True, exist_ok=True)
    (out / "feature_validity_manifest.json").write_text(
        json.dumps(
            {
                "task_id": "0608T004",
                "schema_version": "canonical_feature_conditioned_signal_validity_v1",
                "assessed_regime_id": momentum.TARGET_REGIME_ID,
                "final_recommendation": (
                    "watch_needs_contract_visibility_clarification" if valid else "close_regime_011_line"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        out / "feature_pattern_validity_summary.csv",
        [
            {
                "feature": "context_basis_mid_ticks",
                "variant": "sign_positive",
                "signal_validity": "invalid_not_decision_visible",
                "stability_classification": "stable_all_samples",
                "tail_risk_classification": "tail_risk_acceptable_proxy",
            }
        ],
        ["feature", "variant", "signal_validity", "stability_classification", "tail_risk_classification"],
    )
    return out


def _prepare_basis_fixture(base: Path, *, synthetic: bool = False, future_join: bool = False) -> Path:
    input_dir = _aggregate(base, synthetic=synthetic)
    if synthetic:
        return input_dir
    manifest = json.loads((input_dir / "multi_sample_manifest.json").read_text(encoding="utf-8"))
    for sample in manifest["samples"]:
        pricing_path = Path(sample["pricing_signal_rows"])
        pricing_rows = _read_csv(pricing_path)
        source_rows = []
        for row in pricing_rows:
            row["context_basis_mid_ticks"] = "10"
            row["hyperliquid_future_mid_move_ticks"] = "12"
            row["basis_contract_caveat"] = "diagnostic_only_binance_usdm_futures_BTCUSDT_vs_hyperliquid_BTC_contract_basis"
            if future_join:
                row["binance_local_ts"] = str(int(row["hyperliquid_decision_ts"]) + 1)
            source_rows.append(
                {
                    "join_seq": row["source_row_index"],
                    "binance_source_found": "true",
                    "binance_mid_px": "101",
                    "hyperliquid_mid_px": "100",
                    "basis_mid_ticks": "10",
                }
            )
        _write_csv(pricing_path, pricing_rows, list(pricing_rows[0]))
        join_dir = Path(sample["source_join_dir"])
        _write_csv(
            join_dir / "cross_exchange_joined_features.csv",
            source_rows,
            ["join_seq", "binance_source_found", "binance_mid_px", "hyperliquid_mid_px", "basis_mid_ticks"],
        )
    return input_dir


def _build(tmp_path: Path, *, synthetic: bool = False, future_join: bool = False, t004_valid: bool = True) -> dict[str, object]:
    return basis.build_basis_context_visibility(
        input_dir=_prepare_basis_fixture(tmp_path, synthetic=synthetic, future_join=future_join),
        feature_validity_dir=_feature_validity_dir(tmp_path, valid=t004_valid),
        data_contract_dir=_contract_dir(tmp_path),
        output_dir=tmp_path / "out",
    )


def test_accepts_canonical_input_and_writes_artifacts(tmp_path: Path) -> None:
    result = _build(tmp_path)
    manifest = json.loads((tmp_path / "out" / "basis_visibility_manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == basis.SCHEMA_VERSION
    assert manifest["basis_positive_row_count"] == 18
    assert manifest["final_contract_decision"] == "upgrade_to_context_only_supported"
    assert _read_csv(tmp_path / "out" / "basis_lineage_audit.csv")
    assert _read_csv(tmp_path / "out" / "basis_positive_pattern_by_sample.csv")
    assert _read_csv(tmp_path / "out" / "basis_timestamp_join_audit.csv")
    assert _read_csv(tmp_path / "out" / "basis_horizon_persistence.csv")
    assert _read_csv(tmp_path / "out" / "basis_contract_decision.csv")
    assert result["decision_rows"][0]["lineage_classification"] == "lineage_confirmed_decision_time_formula"


def test_refuses_diagnostic_only_synthetic_input(tmp_path: Path) -> None:
    with pytest.raises(loader.EvidenceValidationError):
        _build(tmp_path, synthetic=True)


def test_requires_t004_prerequisite(tmp_path: Path) -> None:
    with pytest.raises(basis.BasisVisibilityInputError):
        _build(tmp_path, t004_valid=False)


def test_future_input_join_keeps_diagnostic_contract(tmp_path: Path) -> None:
    result = _build(tmp_path, future_join=True)
    decision = result["decision_rows"][0]

    assert decision["timestamp_classification"] == "timestamp_failed_future_join_or_missing"
    assert decision["contract_decision"] == "keep_diagnostic_only_contract_caveat"


def test_contract_decision_taxonomy(tmp_path: Path) -> None:
    result = _build(tmp_path)
    decision = result["decision_rows"][0]["contract_decision"]

    assert decision in {
        "upgrade_to_context_only_supported",
        "watch_contract_visibility_needs_more_samples",
        "keep_diagnostic_only_contract_caveat",
        "reject_basis_context",
    }
