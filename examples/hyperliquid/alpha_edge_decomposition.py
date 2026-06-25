#!/usr/bin/env python3
"""Decompose Binance-led Hyperliquid alpha and production edge blockers.

The runner is deliberately offline and evidence-layer aware. Historical
decision-time signal rows, canonical multi-sample summaries, maker proxies, and
the production public-shadow funnel are analyzed separately because they do not
share a row-level clock.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0625T001"
SCHEMA_VERSION = "cross_exchange_mvp_alpha_edge_decomposition_v1"
FEATURES = [
    "binance_top5_imbalance",
    "binance_microprice_minus_mid_ticks",
    "binance_mid_move_ticks_from_prev",
    "binance_top5_bid_qty",
]
HORIZONS_MS = [100, 250, 500, 1000]
EDGE_THRESHOLDS = [0, 1, 2, 3, 5, 7]
DEFAULT_PRICING_DIR = (
    PROJECT_ROOT / "local_live_analysis" / "binance_led_hyperliquid_pricing_signal_0601T005"
)
DEFAULT_CANONICAL_DIR = (
    PROJECT_ROOT / "local_live_analysis" / "event_mode_canonical_pricing_signal_0604T003"
)
DEFAULT_CANONICAL_VALIDATION_DIR = (
    PROJECT_ROOT / "local_live_analysis" / "canonical_event_mode_evidence_0604T004"
)
DEFAULT_MAKER_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_maker_executability_0608T002"
DEFAULT_PRODUCTION_DIR = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "hyperliquid_tiny_live_m2_aws_repaired_public_shadow_funnel_0624T003_20260624T063826Z"
    / "venv_public_shadow_live"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_alpha_edge_decomposition_0625T001"
)
BOUNDARY_FLAGS = {
    "offline_only": True,
    "public_only": True,
    "no_network_collection": True,
    "no_live_orders": True,
    "no_credentials": True,
    "no_private_account_order_cancel_endpoints": True,
    "no_live_client": True,
    "no_remote_refresh_or_final_gate": True,
    "no_quote_distance_or_cap_relaxation": True,
    "no_taker_or_crossing": True,
    "no_default_on_m3_stable_pnl_or_promotion": True,
    "future_labels_not_decision_inputs": True,
}


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _fmt(value: float | None, places: int = 8) -> str:
    if value is None or not math.isfinite(value):
        return ""
    text = f"{value:.{places}f}".rstrip("0").rstrip(".")
    return text or "0"


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    dx = [value - mean_x for value in xs]
    dy = [value - mean_y for value in ys]
    denom = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denom == 0:
        return 0.0
    return sum(x * y for x, y in zip(dx, dy)) / denom


def _required_paths(
    pricing_dir: Path,
    canonical_dir: Path,
    canonical_validation_dir: Path,
    maker_dir: Path,
    production_dir: Path,
) -> dict[str, Path]:
    return {
        "pricing_rows": pricing_dir / "pricing_signal_rows.csv",
        "pricing_manifest": pricing_dir / "run_manifest.json",
        "canonical_stability": canonical_dir / "feature_horizon_stability_across_samples.csv",
        "canonical_manifest": canonical_dir / "multi_sample_manifest.json",
        "canonical_validation": canonical_validation_dir / "canonical_sample_manifest.json",
        "maker_summary": maker_dir / "regime_executability_summary.csv",
        "maker_spread_adverse": maker_dir / "spread_capture_adverse_selection.csv",
        "production_candidates": production_dir / "current_candidate_audit.csv",
        "production_anti_drift": production_dir / "anti_drift_gate_matrix.csv",
        "production_fair_mid": production_dir / "fair_mid_source_matrix.csv",
        "production_edge": production_dir / "edge_gate_matrix.csv",
        "production_manifest": production_dir / "public_shadow_source_manifest.json",
        "production_boundary": production_dir / "boundary_manifest.json",
    }


def _ensure_required(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required accepted artifacts: " + "; ".join(missing))


def _coverage_rows(paths: dict[str, Path], loaded: dict[str, Any]) -> list[dict[str, Any]]:
    requested_alias = PROJECT_ROOT / "local_live_analysis" / "canonical_event_mode_evidence_0604T003"
    rows = [
        {
            "evidence_layer": "historical_single_sample_signal",
            "artifact": str(paths["pricing_rows"]),
            "exists": paths["pricing_rows"].exists(),
            "row_count": len(loaded["pricing_rows"]),
            "decision_time_features": "yes",
            "future_labels": "yes",
            "production_gate_fields": "no",
            "row_level_cross_layer_join_allowed": "no",
            "coverage_note": "one synchronized 30-minute public sample",
        },
        {
            "evidence_layer": "canonical_multi_sample_signal",
            "artifact": str(paths["canonical_stability"]),
            "exists": paths["canonical_stability"].exists(),
            "row_count": len(loaded["canonical_rows"]),
            "decision_time_features": "aggregate",
            "future_labels": "aggregate",
            "production_gate_fields": "no",
            "row_level_cross_layer_join_allowed": "no",
            "coverage_note": "three accepted event-mode samples; aggregate effects only",
        },
        {
            "evidence_layer": "canonical_validation",
            "artifact": str(paths["canonical_validation"]),
            "exists": paths["canonical_validation"].exists(),
            "row_count": len(loaded["canonical_validation"].get("samples", [])),
            "decision_time_features": "metadata",
            "future_labels": "metadata",
            "production_gate_fields": "no",
            "row_level_cross_layer_join_allowed": "no",
            "coverage_note": (
                "task input named canonical_event_mode_evidence_0604T003, "
                f"but accepted local validation is {paths['canonical_validation'].parent.name}; "
                f"requested path exists={requested_alias.exists()}"
            ),
        },
        {
            "evidence_layer": "canonical_maker_proxy",
            "artifact": str(paths["maker_summary"]),
            "exists": paths["maker_summary"].exists(),
            "row_count": len(loaded["maker_rows"]),
            "decision_time_features": "regime aggregate",
            "future_labels": "public proxy",
            "production_gate_fields": "no",
            "row_level_cross_layer_join_allowed": "no",
            "coverage_note": "not real fill, queue, or private lifecycle evidence",
        },
        {
            "evidence_layer": "production_public_shadow",
            "artifact": str(paths["production_candidates"]),
            "exists": paths["production_candidates"].exists(),
            "row_count": len(loaded["candidate_rows"]),
            "decision_time_features": "partial",
            "future_labels": "no",
            "production_gate_fields": "yes",
            "row_level_cross_layer_join_allowed": "no",
            "coverage_note": "anti-drift/fair-mid/edge available; same-window future markout absent",
        },
    ]
    return rows


def _canonical_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    return {
        (row.get("feature", ""), int(row.get("horizon_ms") or 0)): row
        for row in rows
        if row.get("label") == "hyperliquid_future_mid_move_ticks"
    }


def _signal_response_rows(
    pricing_rows: list[dict[str, str]], canonical_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for row in pricing_rows:
        horizon = int(row.get("horizon_ms") or 0)
        if horizon not in HORIZONS_MS:
            continue
        outcome = _float(row.get("hyperliquid_future_mid_move_ticks"))
        if outcome is None:
            continue
        for feature in FEATURES:
            value = _float(row.get(f"input_{feature}_z"))
            if value is not None:
                grouped[(feature, horizon)].append((value, outcome))

    canonical = _canonical_lookup(canonical_rows)
    output: list[dict[str, Any]] = []
    for feature in FEATURES:
        for horizon in HORIZONS_MS:
            pairs = grouped.get((feature, horizon), [])
            xs = [item[0] for item in pairs]
            ys = [item[1] for item in pairs]
            high = [outcome for value, outcome in pairs if value >= 0.5]
            low = [outcome for value, outcome in pairs if value <= -0.5]
            effect = None if not high or not low else statistics.fmean(high) - statistics.fmean(low)
            canonical_row = canonical.get((feature, horizon), {})
            output.append(
                {
                    "feature": feature,
                    "horizon_ms": horizon,
                    "historical_row_count": len(pairs),
                    "historical_pearson_corr": _fmt(_pearson(xs, ys)),
                    "historical_high_z_row_count": len(high),
                    "historical_low_z_row_count": len(low),
                    "historical_high_minus_low_effect_ticks": _fmt(effect),
                    "historical_direction": (
                        "positive" if effect is not None and effect > 0 else "negative" if effect is not None and effect < 0 else "unknown"
                    ),
                    "canonical_sample_count": canonical_row.get("canonical_eligible_sample_count", ""),
                    "canonical_direction": canonical_row.get("majority_direction", ""),
                    "canonical_direction_consistency_ratio": canonical_row.get("direction_consistency_ratio", ""),
                    "canonical_mean_effect_ticks": canonical_row.get("mean_high_minus_low_effect", ""),
                    "canonical_mean_abs_corr": canonical_row.get("mean_abs_corr", ""),
                    "canonical_verdict": canonical_row.get("stability_verdict", ""),
                    "effective_horizon_scope": "future label at or after nominal horizon",
                }
            )
    return output


def _lead_move_rows(
    signal_rows: list[dict[str, Any]],
    fair_rows: list[dict[str, str]],
    edge_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in signal_rows:
        if int(row["horizon_ms"]) != 1000:
            continue
        output.append(
            {
                "evidence_layer": "canonical_historical",
                "event_sequence": "",
                "feature": row["feature"],
                "side": "",
                "source_status": "accepted_aggregate",
                "source_age_ms": "",
                "basis_mid_ticks": "",
                "lead_move_ticks": row["canonical_mean_effect_ticks"],
                "quote_px": "",
                "fair_mid_px": "",
                "edge_ticks": "",
                "direction_or_side_alignment": row["canonical_direction"],
                "magnitude_status": "aggregate_high_minus_low_effect_not_live_projection",
                "future_label_status": "available_in_historical_aggregate",
            }
        )

    fair_by_event = {row.get("event_sequence", ""): row for row in fair_rows}
    for edge in edge_rows:
        fair = fair_by_event.get(edge.get("event_sequence", ""), {})
        side = edge.get("side", "")
        lead_move = _float(fair.get("lead_move_ticks"))
        expected_sign = 1 if side == "buy" else -1 if side == "sell" else 0
        aligned = (
            "aligned"
            if lead_move is not None and expected_sign != 0 and lead_move * expected_sign > 0
            else "zero"
            if lead_move == 0
            else "opposed"
            if lead_move is not None and expected_sign != 0
            else "unavailable"
        )
        output.append(
            {
                "evidence_layer": "production_public_shadow",
                "event_sequence": edge.get("event_sequence", ""),
                "feature": "production_lead_move_ticks",
                "side": side,
                "source_status": fair.get("source_status", "missing"),
                "source_age_ms": fair.get("source_age_ms", ""),
                "basis_mid_ticks": fair.get("basis_mid_ticks", ""),
                "lead_move_ticks": fair.get("lead_move_ticks", ""),
                "quote_px": edge.get("quote_px", ""),
                "fair_mid_px": edge.get("fair_mid_px", ""),
                "edge_ticks": edge.get("edge_ticks", ""),
                "direction_or_side_alignment": aligned,
                "magnitude_status": "live_projection_observed_but_not_outcome_calibrated",
                "future_label_status": "unsupported_missing_same_window_future_labels",
            }
        )
    return output


def _anti_drift_rows(
    rows: list[dict[str, str]], maker_rows: list[dict[str, str]], maker_adverse_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    grouped = Counter((row.get("status", ""), row.get("reason", "")) for row in rows)
    output: list[dict[str, Any]] = []
    for (status, reason), count in sorted(grouped.items()):
        output.append(
            {
                "evidence_layer": "production_public_shadow",
                "anti_drift_status": status,
                "anti_drift_reason": reason,
                "row_count": count,
                "future_markout_ticks": "",
                "markout_status": "unsupported_missing_same_window_future_labels",
                "interpretation": (
                    "gate throughput only; cannot determine whether blocked rows were favorable or adverse"
                ),
            }
        )
    if maker_rows:
        adverse = maker_adverse_rows[0] if maker_adverse_rows else {}
        output.append(
            {
                "evidence_layer": "canonical_maker_proxy",
                "anti_drift_status": "not_present",
                "anti_drift_reason": "separate public proxy",
                "row_count": maker_rows[0].get("public_proxy_row_count", ""),
                "future_markout_ticks": adverse.get("adverse_selection_proxy_ticks", ""),
                "markout_status": "public_proxy_not_joined_to_production_gate",
                "interpretation": maker_rows[0].get("final_recommendation", ""),
            }
        )
    return output


def _edge_sensitivity_rows(edge_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    fresh_edges = [
        value
        for row in edge_rows
        if row.get("edge_gate_reason") != "fair_mid_source_stale"
        and (value := _float(row.get("edge_ticks"))) is not None
    ]
    output: list[dict[str, Any]] = []
    for threshold in EDGE_THRESHOLDS:
        passing = [value for value in fresh_edges if value > threshold]
        output.append(
            {
                "evidence_layer": "production_public_shadow",
                "threshold_ticks": threshold,
                "fresh_edge_row_count": len(fresh_edges),
                "pass_count": len(passing),
                "pass_rate": _fmt(len(passing) / len(fresh_edges) if fresh_edges else None),
                "edge_min": _fmt(min(fresh_edges) if fresh_edges else None),
                "edge_mean": _fmt(_mean(fresh_edges)),
                "edge_max": _fmt(max(fresh_edges) if fresh_edges else None),
                "interpretation": "diagnostic_only_threshold_sensitivity_not_policy_change",
            }
        )
    return output


def _root_causes(
    *,
    signal_rows: list[dict[str, Any]],
    anti_rows: list[dict[str, str]],
    fair_rows: list[dict[str, str]],
    edge_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], str]:
    stable_1000 = [
        row
        for row in signal_rows
        if int(row["horizon_ms"]) == 1000
        and row["canonical_verdict"] == "stable_across_samples"
        and row["canonical_direction_consistency_ratio"] == "1"
    ]
    anti_counts = Counter(row.get("status", "") for row in anti_rows)
    fair_counts = Counter(row.get("source_status", "") for row in fair_rows)
    valid_edges = [
        _float(row.get("edge_ticks"))
        for row in edge_rows
        if row.get("edge_gate_reason") != "fair_mid_source_stale"
    ]
    clean_edges = [value for value in valid_edges if value is not None]
    fair_by_event = {row.get("event_sequence", ""): row for row in fair_rows}
    opposed = 0
    zero = 0
    for edge in edge_rows:
        fair = fair_by_event.get(edge.get("event_sequence", ""), {})
        lead = _float(fair.get("lead_move_ticks"))
        side = edge.get("side", "")
        expected = 1 if side == "buy" else -1 if side == "sell" else 0
        if lead == 0:
            zero += 1
        elif lead is not None and expected and lead * expected < 0:
            opposed += 1

    causes = [
        {
            "rank": 1,
            "root_cause": "production_edge_sample_coverage_insufficient",
            "evidence_count": len(edge_rows),
            "severity": "blocking",
            "finding": "only four production candidates reached fair-mid/edge; one was stale",
            "required_next_evidence": "multi-window same-schema public shadow with future labels",
        },
        {
            "rank": 2,
            "root_cause": "candidate_side_not_aligned_with_observed_lead_move",
            "evidence_count": opposed + zero,
            "severity": "blocking",
            "finding": f"opposed={opposed}, zero={zero}; valid edge values={clean_edges}",
            "required_next_evidence": "record top5-derived signal components and bind maker side to frozen signal contract",
        },
        {
            "rank": 3,
            "root_cause": "anti_drift_throughput_dominates",
            "evidence_count": anti_counts.get("block", 0),
            "severity": "diagnostic_blocking",
            "finding": (
                f"anti-drift block={anti_counts.get('block', 0)}, pass={anti_counts.get('pass', 0)}; "
                "same-window future markout is absent"
            ),
            "required_next_evidence": "future markout for every anti-drift pass/block row",
        },
        {
            "rank": 4,
            "root_cause": "fair_mid_source_freshness",
            "evidence_count": fair_counts.get("block", 0),
            "severity": "secondary",
            "finding": f"fair-mid pass={fair_counts.get('pass', 0)}, block={fair_counts.get('block', 0)}",
            "required_next_evidence": "source age and sequence fields on every candidate",
        },
        {
            "rank": 5,
            "root_cause": "historical_alpha_exists_but_live_projection_is_unfrozen",
            "evidence_count": len(stable_1000),
            "severity": "opportunity",
            "finding": f"{len(stable_1000)}/{len(FEATURES)} allowlist features stable across three samples at 1000ms",
            "required_next_evidence": "frozen composite mapping from top5 features to lead_move_ticks",
        },
    ]
    recommendation = "needs_more_public_samples" if stable_1000 else "reject_current_signal_shape"
    return causes, recommendation


def _write_recommendation(
    path: Path,
    *,
    recommendation: str,
    causes: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# Alpha / Edge Decomposition Recommendation",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "## Recommendation",
        "",
        f"- `{recommendation}`",
        "",
        "## Finding",
        "",
        "- Historical event-mode evidence supports directional Binance top5 alpha across three samples.",
        "- The current production public-shadow signal shape is not ready to freeze: only four rows reached edge, one source was stale, and the fresh rows did not provide edge above the current buffer.",
        "- Production anti-drift rows have no same-window future markout, so the gate cannot yet be classified as helpful or over-filtering.",
        "- This result requests more synchronized public evidence; it does not lower the seven-tick edge policy.",
        "",
        "## Root Causes",
        "",
    ]
    for row in causes:
        lines.append(f"{row['rank']}. `{row['root_cause']}`: {row['finding']}")
    req = manifest["next_sample_requirements"]
    lines.extend(
        [
            "",
            "## Required T002 Evidence",
            "",
            f"- Separated windows: at least `{req['minimum_windows']}`",
            f"- Duration per window: at least `{req['minimum_minutes_per_window']}` minutes",
            f"- Edge-evaluable rows: at least `{req['minimum_edge_rows_aggregate']}` aggregate and "
            f"`{req['minimum_edge_rows_per_window']}` per window",
            f"- Regimes: at least `{req['minimum_regime_count']}` distinct volatility/liquidity regimes",
            "- Required on every decision: dual top5, local/exchange timestamps, source seq/age, signal components, "
            "lead_move_ticks, candidate side/quote, anti-drift result, fair-mid, edge, and future HL mid/microprice labels.",
            "",
            "## Boundary",
            "",
            "- Offline/public-only/no-submit.",
            "- No live behavior change, credentials, private/order endpoints, quote relaxation, canary, M3, stable-PnL, default-on, or promotion authorization.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_artifacts(
    *,
    pricing_dir: str | Path = DEFAULT_PRICING_DIR,
    canonical_dir: str | Path = DEFAULT_CANONICAL_DIR,
    canonical_validation_dir: str | Path = DEFAULT_CANONICAL_VALIDATION_DIR,
    maker_dir: str | Path = DEFAULT_MAKER_DIR,
    production_dir: str | Path = DEFAULT_PRODUCTION_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    pricing = _expand(pricing_dir)
    canonical = _expand(canonical_dir)
    validation = _expand(canonical_validation_dir)
    maker = _expand(maker_dir)
    production = _expand(production_dir)
    output = _expand(output_dir)
    paths = _required_paths(pricing, canonical, validation, maker, production)
    _ensure_required(paths)

    loaded = {
        "pricing_rows": _read_csv(paths["pricing_rows"]),
        "pricing_manifest": _read_json(paths["pricing_manifest"]),
        "canonical_rows": _read_csv(paths["canonical_stability"]),
        "canonical_manifest": _read_json(paths["canonical_manifest"]),
        "canonical_validation": _read_json(paths["canonical_validation"]),
        "maker_rows": _read_csv(paths["maker_summary"]),
        "maker_adverse_rows": _read_csv(paths["maker_spread_adverse"]),
        "candidate_rows": _read_csv(paths["production_candidates"]),
        "anti_rows": _read_csv(paths["production_anti_drift"]),
        "fair_rows": _read_csv(paths["production_fair_mid"]),
        "edge_rows": _read_csv(paths["production_edge"]),
        "production_manifest": _read_json(paths["production_manifest"]),
        "production_boundary": _read_json(paths["production_boundary"]),
    }

    coverage = _coverage_rows(paths, loaded)
    signal = _signal_response_rows(loaded["pricing_rows"], loaded["canonical_rows"])
    lead_move = _lead_move_rows(signal, loaded["fair_rows"], loaded["edge_rows"])
    anti = _anti_drift_rows(loaded["anti_rows"], loaded["maker_rows"], loaded["maker_adverse_rows"])
    edge = _edge_sensitivity_rows(loaded["edge_rows"])
    causes, recommendation = _root_causes(
        signal_rows=signal,
        anti_rows=loaded["anti_rows"],
        fair_rows=loaded["fair_rows"],
        edge_rows=loaded["edge_rows"],
    )

    next_requirements = {
        "minimum_windows": 3,
        "minimum_minutes_per_window": 30,
        "minimum_edge_rows_aggregate": 100,
        "minimum_edge_rows_per_window": 20,
        "minimum_regime_count": 2,
        "required_fields": [
            "binance_top5_prices_qtys_imbalance_microprice",
            "hyperliquid_top5_prices_qtys_order_counts_microprice",
            "local_receive_and_exchange_timestamps",
            "source_sequence_and_age",
            "signal_components_and_composite",
            "lead_move_ticks",
            "candidate_side_quote_and_spread",
            "anti_drift_status_reason",
            "fair_mid_and_edge_ticks",
            "future_hl_mid_and_microprice_at_100_250_500_1000ms",
        ],
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "recommendation": recommendation,
        "evidence_policy": "layers_are_not_row_joined_without_shared_clock",
        "timestamp_policy": {
            "historical_join": "binance_local_ts <= hyperliquid_decision_ts",
            "future_labels": "first Hyperliquid row at or after nominal horizon",
            "future_labels_are_decision_inputs": False,
        },
        "input_paths": {name: str(path) for name, path in paths.items()},
        "row_counts": {
            "historical_pricing_rows": len(loaded["pricing_rows"]),
            "canonical_stability_rows": len(loaded["canonical_rows"]),
            "canonical_sample_count": len(loaded["canonical_validation"].get("samples", [])),
            "production_candidate_rows": len(loaded["candidate_rows"]),
            "production_anti_drift_rows": len(loaded["anti_rows"]),
            "production_fair_mid_rows": len(loaded["fair_rows"]),
            "production_edge_rows": len(loaded["edge_rows"]),
        },
        "join_quality": loaded["pricing_manifest"].get("quality", {}).get("source_join_quality", {}),
        "production_funnel": {
            "fresh_touch_allowed_count": loaded["production_manifest"].get(
                "fresh_touch_allowed_count", len(loaded["anti_rows"])
            ),
            "anti_drift_pass_count": loaded["production_manifest"].get("anti_drift_pass_count", 0),
            "anti_drift_block_count": loaded["production_manifest"].get("anti_drift_block_count", 0),
            "fair_mid_source_pass_count": loaded["production_manifest"].get("fair_mid_source_pass_count", 0),
            "fair_mid_source_block_count": loaded["production_manifest"].get("fair_mid_source_block_count", 0),
            "edge_gate_pass_count": loaded["production_manifest"].get("edge_gate_pass_count", 0),
            "edge_gate_block_count": loaded["production_manifest"].get("edge_gate_block_count", 0),
        },
        "next_sample_requirements": next_requirements,
        "boundary_flags": BOUNDARY_FLAGS,
        "artifacts": {
            "manifest": str(output / "alpha_edge_decomposition_manifest.json"),
            "source_coverage": str(output / "source_coverage_matrix.csv"),
            "signal_response": str(output / "signal_response_by_horizon.csv"),
            "lead_move": str(output / "lead_move_calibration.csv"),
            "anti_drift": str(output / "anti_drift_markout_interaction.csv"),
            "edge_sensitivity": str(output / "edge_buffer_sensitivity.csv"),
            "root_causes": str(output / "root_cause_summary.csv"),
            "recommendation": str(output / "recommendation.md"),
            "boundary": str(output / "boundary_manifest.json"),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output / "source_coverage_matrix.csv",
        coverage,
        [
            "evidence_layer",
            "artifact",
            "exists",
            "row_count",
            "decision_time_features",
            "future_labels",
            "production_gate_fields",
            "row_level_cross_layer_join_allowed",
            "coverage_note",
        ],
    )
    _write_csv(
        output / "signal_response_by_horizon.csv",
        signal,
        list(signal[0]) if signal else [],
    )
    _write_csv(
        output / "lead_move_calibration.csv",
        lead_move,
        list(lead_move[0]) if lead_move else [],
    )
    _write_csv(
        output / "anti_drift_markout_interaction.csv",
        anti,
        list(anti[0]) if anti else [],
    )
    _write_csv(
        output / "edge_buffer_sensitivity.csv",
        edge,
        list(edge[0]) if edge else [],
    )
    _write_csv(
        output / "root_cause_summary.csv",
        causes,
        list(causes[0]) if causes else [],
    )
    _write_json(output / "alpha_edge_decomposition_manifest.json", manifest)
    _write_json(
        output / "boundary_manifest.json",
        {
            "task_id": TASK_ID,
            "recommendation": recommendation,
            "boundary_flags": BOUNDARY_FLAGS,
            "production_source_boundary": loaded["production_boundary"],
        },
    )
    _write_recommendation(output / "recommendation.md", recommendation=recommendation, causes=causes, manifest=manifest)
    return {
        "manifest": manifest,
        "coverage_rows": coverage,
        "signal_rows": signal,
        "lead_move_rows": lead_move,
        "anti_drift_rows": anti,
        "edge_rows": edge,
        "root_cause_rows": causes,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pricing-dir", default=str(DEFAULT_PRICING_DIR))
    parser.add_argument("--canonical-dir", default=str(DEFAULT_CANONICAL_DIR))
    parser.add_argument("--canonical-validation-dir", default=str(DEFAULT_CANONICAL_VALIDATION_DIR))
    parser.add_argument("--maker-dir", default=str(DEFAULT_MAKER_DIR))
    parser.add_argument("--production-dir", default=str(DEFAULT_PRODUCTION_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_artifacts(
        pricing_dir=args.pricing_dir,
        canonical_dir=args.canonical_dir,
        canonical_validation_dir=args.canonical_validation_dir,
        maker_dir=args.maker_dir,
        production_dir=args.production_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
