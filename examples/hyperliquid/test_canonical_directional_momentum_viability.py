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


PRICING_FIELDS = [
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
    "input_binance_microprice_minus_mid_ticks",
    "input_binance_mid_move_ticks_from_prev",
    "input_binance_top5_bid_qty",
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
]


def _pricing_rows(sample_id: str, *, weak: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(6):
        future_move = 10.0 if not weak else (0.5 if index % 2 == 0 else -0.5)
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
                "input_binance_microprice_minus_mid_ticks": 0.5,
                "input_binance_mid_move_ticks_from_prev": 2.0,
                "input_binance_top5_bid_qty": 10,
                "context_hyperliquid_mid_px": 100,
                "context_hyperliquid_spread_ticks": 15,
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
    for horizon in (100, 250, 5000, 10000):
        extra = dict(rows[0])
        extra["horizon_ms"] = horizon
        rows.append(extra)
    return rows


def _sample(base: Path, sample_id: str, *, synthetic: bool = False, weak: bool = False) -> dict[str, object]:
    pricing_dir = base / sample_id / "pricing"
    status = loader.DIAGNOSTIC_SYNTHETIC_STATUS if synthetic else loader.CANONICAL_EVENT_STATUS
    mode = "synthetic_fixed_grid" if synthetic else "event"
    rows = [] if synthetic else _pricing_rows(sample_id, weak=weak)
    _write_csv(pricing_dir / "pricing_signal_rows.csv", rows, PRICING_FIELDS)
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
        "event_decision_count": 10 if not synthetic else 0,
        "synthetic_decision_count": 0 if not synthetic else 10,
        "run_manifest": str(pricing_dir / "run_manifest.json"),
        "pricing_signal_rows": str(pricing_dir / "pricing_signal_rows.csv"),
    }


def _aggregate(base: Path, *, synthetic: bool = False, weak: bool = False) -> Path:
    out = base / "aggregate"
    samples = [_sample(base, f"s{index}", synthetic=synthetic, weak=weak) for index in range(1, 4)]
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
        [
            {
                "sample_id": sample["sample_id"],
                "pricing_signal_dir": sample["pricing_signal_dir"],
                "source_join_dir": sample["source_join_dir"],
                "source_analysis_dir": sample["source_analysis_dir"],
                "source_sample_dir": sample["source_sample_dir"],
                "decision_mode": sample["decision_mode"],
                "canonical_status": sample["canonical_status"],
                "input_rows": 60,
                "primary_rows": 60,
                "excluded_rows": 0,
                "pricing_signal_rows": 10,
                "future_join_count": 0,
                "missing_binance_join_count": 0,
                "primary_usable_row_count": 60,
                "horizon_count": 4,
                "independent_future_row_delta_count": 3,
            }
            for sample in samples
        ],
        [
            "sample_id",
            "pricing_signal_dir",
            "source_join_dir",
            "source_analysis_dir",
            "source_sample_dir",
            "decision_mode",
            "canonical_status",
            "input_rows",
            "primary_rows",
            "excluded_rows",
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
                "feature": momentum.TARGET_ANCHOR_FEATURE,
                "horizon_ms": 1000,
                "label": "hyperliquid_future_mid_move_ticks",
                "sample_count": 3,
                "eligible_sample_count": 3,
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
                "row_count": 6,
                "effective_future_age_ms_mean": 1000,
                "effective_future_row_delta_mean": 3,
                "distinct_future_row_delta_count": 3,
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
                "row_count": 18 if canonical_count else 0,
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


def _candidate_dir(base: Path) -> Path:
    out = base / "candidate"
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(
        out / "candidate_regime_definitions.csv",
        [
            {
                "regime_id": momentum.TARGET_REGIME_ID,
                "classification": "candidate_for_milestone3_executability",
                "primary_anchor_feature": momentum.TARGET_ANCHOR_FEATURE,
                "horizon_ms": 1000,
                "hyperliquid_context_quality": "primary_usable",
                "hyperliquid_join_age_bucket": "fresh_0_50ms",
                "hyperliquid_spread_bucket": "spread_10_20_ticks",
                "row_count": 18,
                "sample_count": 3,
            }
        ],
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
        json.dumps({"task_id": "0604T009", "schema_version": "canonical_regime_synthesis_v1"}) + "\n",
        encoding="utf-8",
    )
    return out


def _maker_dir(base: Path, *, final: str = "reject_not_maker_executable") -> Path:
    out = base / "maker"
    out.mkdir(parents=True, exist_ok=True)
    (out / "maker_executability_manifest.json").write_text(
        json.dumps(
            {
                "task_id": "0608T002",
                "schema_version": "canonical_maker_executability_v1",
                "assessed_candidate_id": momentum.TARGET_REGIME_ID,
                "final_recommendation": final,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        out / "regime_executability_summary.csv",
        [
            {
                "regime_id": momentum.TARGET_REGIME_ID,
                "final_recommendation": final,
            }
        ],
        ["regime_id", "final_recommendation"],
    )
    return out


def _prereq_dirs(base: Path) -> tuple[Path, Path]:
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
                "diagnostic_rejection_count": 0,
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
                "diagnostic_rejection_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return signal, horizon


def _build(tmp_path: Path, *, weak: bool = False, maker_final: str = "reject_not_maker_executable") -> dict[str, object]:
    input_dir = _aggregate(tmp_path, weak=weak)
    candidate_dir = _candidate_dir(tmp_path)
    maker_dir = _maker_dir(tmp_path, final=maker_final)
    signal_dir, horizon_dir = _prereq_dirs(tmp_path)
    return momentum.build_directional_momentum_viability(
        input_dir=input_dir,
        candidate_dir=candidate_dir,
        maker_executability_dir=maker_dir,
        signal_ranking_dir=signal_dir,
        horizon_regime_dir=horizon_dir,
        output_dir=tmp_path / "out",
    )


def test_accepts_canonical_input_and_uses_manifest_pricing_rows(tmp_path: Path) -> None:
    result = _build(tmp_path)
    output_dir = tmp_path / "out"
    manifest = json.loads((output_dir / "directional_momentum_manifest.json").read_text(encoding="utf-8"))
    base = _read_csv(output_dir / "base_regime_directionality.csv")

    assert manifest["schema_version"] == momentum.SCHEMA_VERSION
    assert manifest["assessed_regime_id"] == momentum.TARGET_REGIME_ID
    assert manifest["row_level_input_policy"] == "multi_sample_manifest.samples[].pricing_signal_rows"
    assert manifest["maker_prerequisite_final_recommendation"] == "reject_not_maker_executable"
    assert base[0]["row_count"] == "18"
    assert _read_csv(output_dir / "feature_directionality_summary.csv")
    assert _read_csv(output_dir / "cost_latency_adjusted_edge.csv")
    assert _read_csv(output_dir / "tail_risk_summary.csv")
    assert _read_csv(output_dir / "directional_candidate_watch_reject.csv")
    assert result["manifest"]["boundary_flags"]["no_order_endpoints"] is True


def test_refuses_diagnostic_only_synthetic_input(tmp_path: Path) -> None:
    input_dir = _aggregate(tmp_path, synthetic=True)
    candidate_dir = _candidate_dir(tmp_path)
    maker_dir = _maker_dir(tmp_path)
    signal_dir, horizon_dir = _prereq_dirs(tmp_path)

    with pytest.raises(loader.EvidenceValidationError):
        momentum.build_directional_momentum_viability(
            input_dir=input_dir,
            candidate_dir=candidate_dir,
            maker_executability_dir=maker_dir,
            signal_ranking_dir=signal_dir,
            horizon_regime_dir=horizon_dir,
            output_dir=tmp_path / "out",
        )


def test_requires_t002_maker_rejection_prerequisite(tmp_path: Path) -> None:
    with pytest.raises(momentum.DirectionalMomentumInputError):
        _build(tmp_path, maker_final="candidate_for_case_library")


def test_feature_conditioned_directionality_and_cost_taxonomy(tmp_path: Path) -> None:
    _build(tmp_path)
    features = _read_csv(tmp_path / "out" / "feature_directionality_summary.csv")
    cost = _read_csv(tmp_path / "out" / "cost_latency_adjusted_edge.csv")[0]

    assert any(
        row["feature"] == "input_binance_mid_move_ticks_from_prev"
        and row["variant"] == "sign_positive"
        and row["classification"] == "directional_signal_supported"
        for row in features
    )
    assert cost["cost_adjusted_viability"] in momentum.COST_CLASSIFICATIONS
    assert cost["cost_assumption_policy"] == "fixed_conservative_proxy_not_optimized"


def test_weak_edge_reject_path(tmp_path: Path) -> None:
    result = _build(tmp_path, weak=True)
    final = result["final_rows"][0]

    assert final["final_recommendation"] in {
        "reject_directional_edge_unstable",
        "reject_net_edge_negative",
    }
