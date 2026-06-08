from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import canonical_basis_positive_robustness as robust
import canonical_event_mode_evidence as loader
from test_canonical_directional_momentum_viability import PRICING_FIELDS, _aggregate, _read_csv, _write_csv


def _contract_dir(base: Path, *, allowed: bool = True) -> Path:
    out = base / "contract"
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(
        out / "feature_decision_table.csv",
        [
            {
                "feature": "basis_mid_dislocation",
                "venue": "Joint",
                "role": "basis_context",
                "decision": "allow" if allowed else "diagnostic_only",
                "status": "context_only_supported" if allowed else "diagnostic_context",
            }
        ],
        ["feature", "venue", "role", "decision", "status"],
    )
    return out


def _basis_visibility_dir(base: Path, *, decision: str = "upgrade_to_context_only_supported") -> Path:
    out = base / "basis_visibility"
    out.mkdir(parents=True, exist_ok=True)
    (out / "basis_visibility_manifest.json").write_text(
        json.dumps(
            {
                "task_id": "0608T005",
                "schema_version": "canonical_basis_context_visibility_v1",
                "final_contract_decision": decision,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def _prepare_input(base: Path, *, synthetic: bool = False, concentrated: bool = False, redundant: bool = False, weak: bool = False) -> Path:
    input_dir = _aggregate(base, synthetic=synthetic)
    if synthetic:
        return input_dir
    manifest = json.loads((input_dir / "multi_sample_manifest.json").read_text(encoding="utf-8"))
    for sample_index, sample in enumerate(manifest["samples"]):
        path = Path(sample["pricing_signal_rows"])
        rows = _read_csv(path)
        updated = []
        for row_index, row in enumerate(rows):
            if concentrated and sample_index > 0:
                row["context_basis_mid_ticks"] = "-1"
            elif redundant:
                row["context_basis_mid_ticks"] = str(10 + row_index)
            else:
                row["context_basis_mid_ticks"] = "10"
            row["context_hyperliquid_spread_ticks"] = str(5 + 10 * (row_index % 3))
            row["context_hyperliquid_join_age_ms"] = str(10 + 60 * (row_index % 2))
            row["input_binance_mid_move_ticks_from_prev"] = str(row_index - 2)
            row["context_hyperliquid_top5_imbalance"] = row["context_basis_mid_ticks"] if redundant else str((-1) ** row_index * 0.2)
            row["context_hyperliquid_microprice_minus_mid_ticks"] = row["context_basis_mid_ticks"] if redundant else str((-1) ** (row_index + 1) * 0.3)
            if weak:
                row["hyperliquid_future_mid_move_ticks"] = "-2" if row_index % 2 else "1"
            else:
                row["hyperliquid_future_mid_move_ticks"] = "20"
            updated.append(row)
            if row["horizon_ms"] == "1000":
                duplicate = dict(row)
                duplicate["source_row_index"] = str(int(row["source_row_index"]) + 100)
                duplicate["future_row_index"] = str(int(row["future_row_index"]) + 100)
                updated.append(duplicate)
        _write_csv(path, updated, PRICING_FIELDS)
    return input_dir


def _build(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    return robust.build_basis_positive_robustness(
        input_dir=_prepare_input(tmp_path, **kwargs),
        basis_visibility_dir=_basis_visibility_dir(tmp_path),
        data_contract_dir=_contract_dir(tmp_path),
        output_dir=tmp_path / "out",
    )


def test_accepts_canonical_input_and_writes_artifacts(tmp_path: Path) -> None:
    result = _build(tmp_path)
    manifest = json.loads((tmp_path / "out" / "basis_positive_robustness_manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == robust.SCHEMA_VERSION
    assert manifest["scope_policy"] == "not_limited_to_regime_011"
    assert manifest["t005_final_contract_decision"] == "upgrade_to_context_only_supported"
    assert manifest["contract_basis_mid_status"] == "context_only_supported"
    assert manifest["final_recommendation"] in robust.FINAL_RECOMMENDATIONS
    assert _read_csv(tmp_path / "out" / "basis_positive_overall_summary.csv")
    assert _read_csv(tmp_path / "out" / "basis_positive_by_spread.csv")
    assert _read_csv(tmp_path / "out" / "basis_positive_by_join_age.csv")
    assert _read_csv(tmp_path / "out" / "basis_positive_by_volatility.csv")
    assert _read_csv(tmp_path / "out" / "basis_positive_by_hl_book_state.csv")
    assert _read_csv(tmp_path / "out" / "basis_positive_collinearity.csv")
    assert _read_csv(tmp_path / "out" / "basis_positive_cost_tail.csv")
    assert _read_csv(tmp_path / "out" / "basis_positive_horizon_persistence.csv")
    assert result["manifest"]["boundary_flags"]["no_order_endpoints"] is True


def test_refuses_diagnostic_only_synthetic_input(tmp_path: Path) -> None:
    with pytest.raises(loader.EvidenceValidationError):
        _build(tmp_path, synthetic=True)


def test_requires_t005_prerequisite(tmp_path: Path) -> None:
    with pytest.raises(robust.BasisPositiveRobustnessInputError):
        robust.build_basis_positive_robustness(
            input_dir=_prepare_input(tmp_path),
            basis_visibility_dir=_basis_visibility_dir(tmp_path, decision="keep_diagnostic_only_contract_caveat"),
            data_contract_dir=_contract_dir(tmp_path),
            output_dir=tmp_path / "out",
        )


def test_rejects_when_contract_not_context_supported(tmp_path: Path) -> None:
    with pytest.raises(robust.BasisPositiveRobustnessInputError):
        robust.build_basis_positive_robustness(
            input_dir=_prepare_input(tmp_path),
            basis_visibility_dir=_basis_visibility_dir(tmp_path),
            data_contract_dir=_contract_dir(tmp_path, allowed=False),
            output_dir=tmp_path / "out",
        )


def test_redundant_hl_book_proxy_reject_path(tmp_path: Path) -> None:
    result = _build(tmp_path, redundant=True)
    assert result["manifest"]["final_recommendation"] == "reject"
    assert any(row["proxy_classification"] == "high_redundancy_proxy_risk" for row in result["collinearity_rows"])


def test_concentrated_or_weak_paths_do_not_keep(tmp_path: Path) -> None:
    concentrated = _build(tmp_path / "concentrated", concentrated=True)
    weak = _build(tmp_path / "weak", weak=True)

    assert concentrated["manifest"]["final_recommendation"] in {"needs_more_samples", "reject"}
    assert weak["manifest"]["final_recommendation"] == "reject"
