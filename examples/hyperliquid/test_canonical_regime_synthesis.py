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
import canonical_regime_synthesis as synthesis
import canonical_signal_quality_ranking as ranking
from test_canonical_horizon_regime_diagnostics import _make_aggregate, _sample


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


def _allowlist_csv(path: Path) -> Path:
    rows = [
        {"feature": feature, "decision": "allow", "status": "primary_allowlist"}
        for feature in [synthesis.PRIMARY_ANCHOR_FEATURE, *synthesis.SECONDARY_CONTEXT_FEATURES]
    ]
    _write_csv(path, rows, ["feature", "decision", "status"])
    return path


def _feature_row(
    feature: str,
    horizon_ms: int,
    *,
    label: str = synthesis.TARGET_LABEL,
    stable: bool = True,
    direction: str = "positive",
    consistency: float = 1.0,
    effect: float = 80.0,
    corr: float = 0.1,
    independent_delta: int = 4,
    sample_effects: str = "s1:canonical_event_mode:positive:10|s2:canonical_event_mode:positive:11|s3:canonical_event_mode:positive:12",
) -> dict[str, object]:
    return {
        "feature": feature,
        "horizon_ms": horizon_ms,
        "label": label,
        "sample_count": 3,
        "eligible_sample_count": 3,
        "canonical_eligible_sample_count": 3,
        "canonical_independent_future_row_delta_count": independent_delta,
        "majority_direction": direction,
        "direction_consistency_ratio": consistency,
        "positive_sample_count": 3 if direction == "positive" else 0,
        "negative_sample_count": 3 if direction == "negative" else 0,
        "zero_sample_count": 0,
        "diagnostic_synthetic_sample_count": 0,
        "total_row_count": 300,
        "mean_high_minus_low_effect": effect if direction == "positive" else -effect,
        "std_high_minus_low_effect": 1,
        "mean_abs_corr": corr,
        "sample_effects": sample_effects,
        "stability_verdict": "stable_across_samples" if stable else "unstable_across_samples",
    }


def _replace_feature_rows(path: Path, rows: list[dict[str, object]]) -> None:
    _write_csv(
        path / "feature_horizon_stability_across_samples.csv",
        rows,
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


def _canonical_input(tmp_path: Path, *, horizons: list[int] | None = None, venue_direction: str = "positive") -> Path:
    horizons = horizons or [500, 1000]
    venue_rows = []
    for horizon_ms in horizons:
        venue_rows.append(
            {
                "horizon_ms": horizon_ms,
                "hyperliquid_context_quality": "primary_usable",
                "hyperliquid_join_age_bucket": "fresh_0_50ms",
                "hyperliquid_spread_bucket": "spread_0_10_ticks",
                "sample_count": 3,
                "row_count": 300,
                "mean_future_mid_move_ticks": 2 if venue_direction == "positive" else -2,
                "sample_means": (
                    "s1:1|s2:1.5|s3:2"
                    if venue_direction == "positive"
                    else "s1:-1|s2:-1.5|s3:-2"
                ),
                "conditioning_status": "multi_sample_regime",
            }
        )
    input_dir = _make_aggregate(
        tmp_path,
        samples=[_sample("s1"), _sample("s2"), _sample("s3")],
        venue_rows=venue_rows,
    )
    feature_rows: list[dict[str, object]] = []
    for feature in [synthesis.PRIMARY_ANCHOR_FEATURE, *synthesis.SECONDARY_CONTEXT_FEATURES]:
        for horizon_ms in horizons:
            stable = feature == synthesis.PRIMARY_ANCHOR_FEATURE
            feature_rows.append(
                _feature_row(
                    feature,
                    horizon_ms,
                    stable=stable,
                    consistency=1.0 if stable else 0.8,
                    effect=100 if feature == synthesis.PRIMARY_ANCHOR_FEATURE else 30,
                    corr=0.2 if feature == synthesis.PRIMARY_ANCHOR_FEATURE else 0.1,
                    independent_delta=4 if horizon_ms >= 1000 else 3,
                )
            )
    _replace_feature_rows(input_dir, feature_rows)
    return input_dir


def _build_dependencies(tmp_path: Path, input_dir: Path, allowlist_csv: Path) -> tuple[Path, Path]:
    signal_dir = tmp_path / "ranking"
    horizon_dir = tmp_path / "diagnostics"
    ranking.build_signal_quality_ranking(
        input_dir=input_dir,
        output_dir=signal_dir,
        allowlist_csv=allowlist_csv,
    )
    diagnostics.build_horizon_regime_diagnostics(input_dir=input_dir, output_dir=horizon_dir)
    return signal_dir, horizon_dir


def _force_anchor_keep(signal_dir: Path) -> None:
    rows = _read_csv(signal_dir / "signal_quality_ranking.csv")
    for row in rows:
        if row["feature"] == synthesis.PRIMARY_ANCHOR_FEATURE:
            row["final_bucket"] = "keep_for_read_only_research"
    _write_csv(signal_dir / "signal_quality_ranking.csv", rows, list(rows[0]))


def test_candidate_requires_mid_move_primary_anchor_and_decision_time_fields(tmp_path: Path) -> None:
    input_dir = _canonical_input(tmp_path, horizons=[1000])
    allowlist = _allowlist_csv(tmp_path / "allowlist.csv")
    signal_dir, horizon_dir = _build_dependencies(tmp_path, input_dir, allowlist)

    result = synthesis.build_canonical_regime_synthesis(
        input_dir=input_dir,
        signal_ranking_dir=signal_dir,
        horizon_regime_dir=horizon_dir,
        output_dir=tmp_path / "synthesis",
        allowlist_csv=allowlist,
    )

    assert result["candidate_rows"]
    for row in result["candidate_rows"]:
        assert row["primary_anchor_feature"] == synthesis.PRIMARY_ANCHOR_FEATURE
        assert row["classification"] == "candidate_for_milestone3_executability"
        assert "hyperliquid_join_age_bucket" in row["decision_time_visible_definition"]
        assert "maker" not in row["decision_time_visible_definition"].lower()


def test_short_horizon_only_remains_watch_not_candidate(tmp_path: Path) -> None:
    input_dir = _canonical_input(tmp_path, horizons=[100])
    allowlist = _allowlist_csv(tmp_path / "allowlist.csv")
    signal_dir, horizon_dir = _build_dependencies(tmp_path, input_dir, allowlist)
    _force_anchor_keep(signal_dir)

    result = synthesis.build_canonical_regime_synthesis(
        input_dir=input_dir,
        signal_ranking_dir=signal_dir,
        horizon_regime_dir=horizon_dir,
        output_dir=tmp_path / "synthesis",
        allowlist_csv=allowlist,
    )

    assert result["candidate_rows"] == []
    assert {row["classification"] for row in result["synthesis_rows"]} == {"watch_needs_more_samples"}
    assert result["synthesis_rows"][0]["reason"] == "100_250ms_short_horizon_watch_only"


def test_direction_disagreement_rejects_regime(tmp_path: Path) -> None:
    input_dir = _canonical_input(tmp_path, horizons=[1000], venue_direction="negative")
    allowlist = _allowlist_csv(tmp_path / "allowlist.csv")
    signal_dir, horizon_dir = _build_dependencies(tmp_path, input_dir, allowlist)

    result = synthesis.build_canonical_regime_synthesis(
        input_dir=input_dir,
        signal_ranking_dir=signal_dir,
        horizon_regime_dir=horizon_dir,
        output_dir=tmp_path / "synthesis",
        allowlist_csv=allowlist,
    )

    assert result["candidate_rows"] == []
    assert result["synthesis_rows"][0]["classification"] == "reject_unstable_direction"
    assert result["synthesis_rows"][0]["reason"] == "anchor_direction_disagrees_with_context_regime_direction"


def test_secondary_features_are_watch_only_in_output(tmp_path: Path) -> None:
    input_dir = _canonical_input(tmp_path, horizons=[1000])
    allowlist = _allowlist_csv(tmp_path / "allowlist.csv")
    signal_dir, horizon_dir = _build_dependencies(tmp_path, input_dir, allowlist)

    result = synthesis.build_canonical_regime_synthesis(
        input_dir=input_dir,
        signal_ranking_dir=signal_dir,
        horizon_regime_dir=horizon_dir,
        output_dir=tmp_path / "synthesis",
        allowlist_csv=allowlist,
    )

    secondary_rows = [
        row for row in result["watch_reject_rows"] if row["item_type"] == "secondary_context_feature"
    ]
    assert {row["item_key"] for row in secondary_rows} == set(synthesis.SECONDARY_CONTEXT_FEATURES)
    assert all(row["boundary"] == "not_allowed_as_primary_anchor_for_promoted_candidate" for row in secondary_rows)


def test_refuses_diagnostic_only_synthetic_input(tmp_path: Path) -> None:
    input_dir = _make_aggregate(
        tmp_path,
        samples=[
            _sample(
                "synth",
                decision_mode="synthetic_fixed_grid",
                canonical_status=loader.DIAGNOSTIC_SYNTHETIC_STATUS,
            )
        ],
        venue_rows=[],
    )
    allowlist = _allowlist_csv(tmp_path / "allowlist.csv")
    signal_dir = tmp_path / "ranking"
    horizon_dir = tmp_path / "diagnostics"
    signal_dir.mkdir()
    horizon_dir.mkdir()
    (signal_dir / "signal_quality_ranking_manifest.json").write_text(
        json.dumps(
            {
                "task_id": "0604T006",
                "schema_version": "canonical_signal_quality_ranking_v1",
                "canonical_sample_count": 0,
                "diagnostic_rejection_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        signal_dir / "signal_quality_ranking.csv",
        [
            {
                "feature": synthesis.PRIMARY_ANCHOR_FEATURE,
                "final_bucket": "keep_for_read_only_research",
            }
        ],
        ["feature", "final_bucket"],
    )
    (horizon_dir / "horizon_regime_diagnostics_manifest.json").write_text(
        json.dumps(
            {
                "task_id": "0604T007",
                "schema_version": "canonical_horizon_regime_diagnostics_v1",
                "canonical_sample_count": 0,
                "diagnostic_rejection_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(horizon_dir / "horizon_independence_diagnostics.csv", [], ["horizon_ms"])
    _write_csv(horizon_dir / "regime_conditioning_diagnostics.csv", [], ["horizon_ms"])

    with pytest.raises(loader.EvidenceValidationError):
        synthesis.build_canonical_regime_synthesis(
            input_dir=input_dir,
            signal_ranking_dir=signal_dir,
            horizon_regime_dir=horizon_dir,
            output_dir=tmp_path / "synthesis",
            allowlist_csv=allowlist,
        )


def test_writes_expected_artifacts_and_boundary_flags(tmp_path: Path) -> None:
    input_dir = _canonical_input(tmp_path, horizons=[1000])
    allowlist = _allowlist_csv(tmp_path / "allowlist.csv")
    signal_dir, horizon_dir = _build_dependencies(tmp_path, input_dir, allowlist)
    output_dir = tmp_path / "synthesis"

    result = synthesis.build_canonical_regime_synthesis(
        input_dir=input_dir,
        signal_ranking_dir=signal_dir,
        horizon_regime_dir=horizon_dir,
        output_dir=output_dir,
        allowlist_csv=allowlist,
    )

    manifest = json.loads((output_dir / "canonical_regime_synthesis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == synthesis.SCHEMA_VERSION
    assert manifest["boundary_flags"]["no_maker_side_selection"] is True
    assert manifest["candidate_count"] == len(result["candidate_rows"])
    assert _read_csv(output_dir / "candidate_regime_evidence_summary.csv")
    assert _read_csv(output_dir / "candidate_regime_definitions.csv")
    assert _read_csv(output_dir / "candidate_regime_watch_reject_list.csv")
    report = (output_dir / "canonical_regime_synthesis_report.md").read_text(encoding="utf-8").lower()
    assert "no maker side" in report
    assert "quote placement" in report
