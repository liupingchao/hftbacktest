#!/usr/bin/env python3
"""Read-only basis-positive robustness diagnosis outside Regime 011."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import canonical_directional_momentum_viability as momentum
import canonical_event_mode_evidence as canonical_loader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0608T006"
SCHEMA_VERSION = "canonical_basis_positive_robustness_v1"
PRIMARY_HORIZON_MS = 1000
DIAGNOSTIC_HORIZONS = [5000, 10000]
WATCH_HORIZONS = [100, 250]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "event_mode_canonical_pricing_signal_0604T003"
DEFAULT_BASIS_VISIBILITY_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_basis_context_visibility_0608T005"
DEFAULT_DATA_CONTRACT_DIR = PROJECT_ROOT / "local_live_analysis" / "binance_led_hyperliquid_data_contract_0601T004"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_basis_positive_robustness_0608T006"

FEE_PROXY_TICKS = 2.0
SLIPPAGE_PROXY_TICKS = 2.0
LATENCY_DECAY_PROXY_TICKS = 5.0
TOTAL_COST_PROXY_TICKS = FEE_PROXY_TICKS + SLIPPAGE_PROXY_TICKS + LATENCY_DECAY_PROXY_TICKS
FINAL_RECOMMENDATIONS = {
    "keep_for_read_only_context_research",
    "needs_more_samples",
    "reject",
}
BOUNDARY_FLAGS = {
    "no_new_data_collection": True,
    "read_only_basis_positive_robustness_diagnosis_only": True,
    "no_executable_trading_instruction": True,
    "no_actual_order_side_output": True,
    "no_quote_price_or_size_output": True,
    "no_leverage_output": True,
    "no_stop_or_take_profit_rule": True,
    "no_deployment_recommendation": True,
    "no_case_library_implementation": True,
    "no_case_library_trigger": True,
    "no_shadow_decision_generation": True,
    "no_private_keys": True,
    "no_private_account_endpoints": True,
    "no_order_endpoints": True,
    "no_order_lifecycle": True,
    "no_strategy_implementation": True,
    "no_live_trading_bot": True,
    "no_parameter_search": True,
    "no_default_on": True,
    "no_tiny_live": True,
    "no_promotion": True,
    "no_schema_or_connector_or_core_api_change": True,
}


class BasisPositiveRobustnessInputError(ValueError):
    """Raised when T006 inputs violate the read-only robustness contract."""


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BasisPositiveRobustnessInputError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _as_float(value: Any) -> float | None:
    return momentum._as_float(value)


def _as_int(value: Any, default: int = 0) -> int:
    return momentum._as_int(value, default=default)


def _fmt(value: float | None, places: int = 8) -> str:
    return momentum._format_float(value, places=places)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], pct: float) -> float | None:
    return momentum._percentile(values, pct)


def _sign(value: float | None) -> int:
    return momentum._sign(value)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _sample_concentration(rows: list[dict[str, str]]) -> float | None:
    counts = Counter(row.get("sample_id", "") for row in rows)
    total = sum(counts.values())
    return max(counts.values()) / total if total else None


def _spread_bucket(value: float | None) -> str:
    if value is None:
        return "spread_missing"
    if value < 10:
        return "spread_lt_10_ticks"
    if value <= 20:
        return "spread_10_20_ticks"
    return "spread_gt_20_ticks"


def _join_age_bucket(value: float | None) -> str:
    if value is None:
        return "join_age_missing"
    if value <= 50:
        return "join_age_0_50ms"
    if value <= 100:
        return "join_age_50_100ms"
    return "join_age_gt_100ms"


def _volatility_bucket(value: float | None, low: float | None, high: float | None) -> str:
    if value is None:
        return "volatility_missing"
    if low is None or high is None or low == high:
        return "volatility_unbucketed"
    if value <= low:
        return "volatility_low"
    if value >= high:
        return "volatility_high"
    return "volatility_mid"


def _signed_bucket(value: float | None, prefix: str) -> str:
    if value is None:
        return f"{prefix}_missing"
    if value > 0:
        return f"{prefix}_positive"
    if value < 0:
        return f"{prefix}_negative"
    return f"{prefix}_zero"


def _load_horizon_rows(loaded: dict[str, Any], horizon_ms: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sample in loaded["source_manifest"].get("samples", []):
        if not isinstance(sample, dict):
            continue
        if sample.get("decision_mode") != "event" or sample.get("canonical_status") != "canonical_event_mode":
            raise BasisPositiveRobustnessInputError("formal input contains non-canonical sample in manifest")
        pricing_path = Path(str(sample.get("pricing_signal_rows", "")))
        if not pricing_path.exists():
            raise BasisPositiveRobustnessInputError(
                f"missing manifest samples[].pricing_signal_rows for sample {sample.get('sample_id')}: {pricing_path}"
            )
        for row in _read_csv(pricing_path):
            if _as_int(row.get("horizon_ms")) == horizon_ms:
                rows.append(row)
    return rows


def _primary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("joined_row_quality") == "primary_usable"
        and row.get("context_hyperliquid_context_quality") == "primary_usable"
        and (_as_float(row.get("context_basis_mid_ticks")) or 0.0) > 0
        and _as_float(row.get("hyperliquid_future_mid_move_ticks")) is not None
    ]


def _validate_t005(basis_visibility_dir: Path) -> dict[str, Any]:
    manifest = _read_json(basis_visibility_dir / "basis_visibility_manifest.json")
    if manifest.get("task_id") != "0608T005":
        raise BasisPositiveRobustnessInputError("basis visibility manifest must be from 0608T005")
    if manifest.get("schema_version") != "canonical_basis_context_visibility_v1":
        raise BasisPositiveRobustnessInputError("unexpected T005 schema")
    if manifest.get("final_contract_decision") != "upgrade_to_context_only_supported":
        raise BasisPositiveRobustnessInputError("T006 requires final_contract_decision=upgrade_to_context_only_supported")
    return manifest


def _validate_contract(data_contract_dir: Path) -> dict[str, str]:
    rows = _read_csv(data_contract_dir / "feature_decision_table.csv")
    matches = [row for row in rows if row.get("feature") == "basis_mid_dislocation"]
    if len(matches) != 1:
        raise BasisPositiveRobustnessInputError("contract must contain one basis_mid_dislocation row")
    row = matches[0]
    if row.get("decision") != "allow" or row.get("status") != "context_only_supported":
        raise BasisPositiveRobustnessInputError("basis_mid_dislocation must be allow/context_only_supported")
    return row


def _stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    moves = [value for value in (_as_float(row.get("hyperliquid_future_mid_move_ticks")) for row in rows) if value is not None]
    positive = sum(1 for value in moves if value > 0)
    negative = sum(1 for value in moves if value < 0)
    nonzero = positive + negative
    sample_edges: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _as_float(row.get("hyperliquid_future_mid_move_ticks"))
        if value is not None:
            sample_edges[row.get("sample_id", "")].append(value)
    per_sample_mean = {sample: _mean(values) for sample, values in sample_edges.items() if values}
    return {
        "row_count": len(rows),
        "sample_count": len(per_sample_mean),
        "positive_future_count": positive,
        "negative_future_count": negative,
        "direction_hit_rate": positive / nonzero if nonzero else None,
        "mean_future_mid_move_ticks": _mean(moves),
        "median_future_mid_move_ticks": statistics.median(moves) if moves else None,
        "p05_future_mid_move_ticks": _percentile(moves, 0.05),
        "p95_future_mid_move_ticks": _percentile(moves, 0.95),
        "mean_abs_future_mid_move_ticks": _mean([abs(value) for value in moves]),
        "sample_direction_consistency": (
            sum(1 for value in per_sample_mean.values() if value is not None and value > 0) / len(per_sample_mean)
            if per_sample_mean
            else None
        ),
        "max_sample_row_share": _sample_concentration(rows),
    }


def _summary_row(group_name: str, group_value: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    stats = _stats(rows)
    return {
        "group_name": group_name,
        "group_value": group_value,
        "row_count": stats["row_count"],
        "sample_count": stats["sample_count"],
        "positive_future_count": stats["positive_future_count"],
        "negative_future_count": stats["negative_future_count"],
        "direction_hit_rate": _fmt(stats["direction_hit_rate"]),
        "mean_future_mid_move_ticks": _fmt(stats["mean_future_mid_move_ticks"]),
        "median_future_mid_move_ticks": _fmt(stats["median_future_mid_move_ticks"]),
        "p05_future_mid_move_ticks": _fmt(stats["p05_future_mid_move_ticks"]),
        "p95_future_mid_move_ticks": _fmt(stats["p95_future_mid_move_ticks"]),
        "sample_direction_consistency": _fmt(stats["sample_direction_consistency"]),
        "max_sample_row_share": _fmt(stats["max_sample_row_share"]),
    }


def _group_summary(rows: list[dict[str, str]], group_name: str, key_fn: Any) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    return [_summary_row(group_name, key, value) for key, value in sorted(groups.items())]


def _volatility_thresholds(rows: list[dict[str, str]]) -> tuple[float | None, float | None]:
    values = [
        abs(value)
        for value in (_as_float(row.get("input_binance_mid_move_ticks_from_prev")) for row in rows)
        if value is not None
    ]
    return _percentile(values, 0.33), _percentile(values, 0.67)


def _collinearity_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    basis = [(_as_float(row.get("context_basis_mid_ticks")), row) for row in rows]
    out: list[dict[str, Any]] = []
    for feature in ["context_hyperliquid_top5_imbalance", "context_hyperliquid_microprice_minus_mid_ticks"]:
        pairs: list[tuple[float, float]] = []
        same_positive = 0
        conflicts = 0
        for basis_value, row in basis:
            feature_value = _as_float(row.get(feature))
            if basis_value is None or feature_value is None:
                continue
            pairs.append((basis_value, feature_value))
            if feature_value > 0:
                same_positive += 1
            elif feature_value < 0:
                conflicts += 1
        xs = [pair[0] for pair in pairs]
        ys = [pair[1] for pair in pairs]
        pearson = _pearson(xs, ys)
        spearman = _spearman(xs, ys)
        abs_corr = max(abs(pearson or 0.0), abs(spearman or 0.0))
        if len(pairs) < 10:
            classification = "insufficient_overlap"
        elif abs_corr >= 0.95:
            classification = "high_redundancy_proxy_risk"
        elif abs_corr >= 0.75:
            classification = "moderate_redundancy_watch"
        else:
            classification = "not_explained_solely_by_hl_book_state"
        out.append(
            {
                "feature": feature,
                "row_count": len(pairs),
                "pearson_corr": _fmt(pearson),
                "spearman_corr": _fmt(spearman),
                "same_positive_overlap_rate": _fmt(same_positive / len(pairs) if pairs else None),
                "sign_conflict_rate": _fmt(conflicts / len(pairs) if pairs else None),
                "proxy_classification": classification,
            }
        )
    return out


def _cost_tail_row(rows: list[dict[str, str]]) -> dict[str, Any]:
    stats = _stats(rows)
    gross = stats["mean_future_mid_move_ticks"]
    net = None if gross is None else gross - TOTAL_COST_PROXY_TICKS
    wrong_way = [
        abs(value)
        for value in (_as_float(row.get("hyperliquid_future_mid_move_ticks")) for row in rows)
        if value is not None and value < 0
    ]
    all_moves = [value for value in (_as_float(row.get("hyperliquid_future_mid_move_ticks")) for row in rows) if value is not None]
    wrong_rate = len(wrong_way) / len(all_moves) if all_moves else None
    p95_loss = _percentile(wrong_way, 0.95)
    max_loss = max(wrong_way) if wrong_way else None
    if net is None or net <= 0:
        cost_class = "cost_tail_reject"
    elif wrong_rate is not None and wrong_rate > 0.45:
        cost_class = "cost_tail_reject"
    elif p95_loss is not None and p95_loss > 80:
        cost_class = "cost_tail_reject"
    elif wrong_rate is not None and wrong_rate > 0.30:
        cost_class = "cost_tail_watch"
    else:
        cost_class = "cost_tail_acceptable_proxy"
    return {
        "pattern": "context_basis_mid_ticks > 0",
        "row_count": len(rows),
        "gross_edge_ticks": _fmt(gross),
        "fee_proxy_ticks": _fmt(FEE_PROXY_TICKS),
        "slippage_proxy_ticks": _fmt(SLIPPAGE_PROXY_TICKS),
        "latency_decay_proxy_ticks": _fmt(LATENCY_DECAY_PROXY_TICKS),
        "net_edge_proxy_ticks": _fmt(net),
        "wrong_way_rate": _fmt(wrong_rate),
        "p95_wrong_way_loss_ticks": _fmt(p95_loss),
        "max_wrong_way_loss_ticks": _fmt(max_loss),
        "cost_tail_classification": cost_class,
        "cost_assumption_policy": "fixed_conservative_proxy_not_optimized",
    }


def _horizon_persistence_rows(loaded: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for horizon in WATCH_HORIZONS + [PRIMARY_HORIZON_MS] + DIAGNOSTIC_HORIZONS:
        rows = _primary_rows(_load_horizon_rows(loaded, horizon))
        stats = _stats(rows)
        out.append(
            {
                "horizon_ms": horizon,
                "evidence_role": "primary" if horizon == PRIMARY_HORIZON_MS else ("watch_alias_context" if horizon in WATCH_HORIZONS else "diagnostic_persistence"),
                "row_count": stats["row_count"],
                "sample_count": stats["sample_count"],
                "direction_hit_rate": _fmt(stats["direction_hit_rate"]),
                "mean_future_mid_move_ticks": _fmt(stats["mean_future_mid_move_ticks"]),
                "direction": "positive" if (stats["mean_future_mid_move_ticks"] or 0.0) > 0 else "non_positive",
            }
        )
    return out


def _final_recommendation(
    *,
    overall: dict[str, Any],
    by_spread: list[dict[str, Any]],
    by_join_age: list[dict[str, Any]],
    by_book: list[dict[str, Any]],
    collinearity: list[dict[str, Any]],
    cost_tail: dict[str, Any],
    horizons: list[dict[str, Any]],
) -> tuple[str, str]:
    if overall["row_count"] < 30 or overall["sample_count"] < 3:
        return "needs_more_samples", "basis-positive evidence has insufficient row or sample coverage"
    if (_as_float(overall["max_sample_row_share"]) or 1.0) > 0.55:
        return "needs_more_samples", "basis-positive evidence is sample-concentrated"
    if any(row["proxy_classification"] == "high_redundancy_proxy_risk" for row in collinearity):
        return "reject", "basis-positive effect is explained by a highly redundant Hyperliquid book-state proxy"
    if cost_tail["cost_tail_classification"] == "cost_tail_reject":
        return "reject", "basis-positive effect fails conservative cost/tail proxy"
    primary_persistent = [
        row for row in horizons if row["horizon_ms"] in {PRIMARY_HORIZON_MS, 5000, 10000}
    ]
    if any(row["direction"] != "positive" for row in primary_persistent):
        return "reject", "basis-positive horizon persistence reverses at 1000/5000/10000ms"
    robust_bucket_count = 0
    for rows in (by_spread, by_join_age, by_book):
        robust_bucket_count += sum(
            1
            for row in rows
            if row["sample_count"] >= 2 and (_as_float(row["mean_future_mid_move_ticks"]) or 0.0) > 0
        )
    if robust_bucket_count < 3:
        return "needs_more_samples", "basis-positive effect is not yet distributed across enough strata"
    return "keep_for_read_only_context_research", "basis-positive context remains multi-sample, stratified, persistent, and not solely a Hyperliquid book-state proxy"


def _write_report(path: Path, manifest: dict[str, Any], recommendation: str, reason: str) -> None:
    lines = [
        "# Basis-Positive Robustness Report",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "## Scope",
        "",
        "- Pattern: `context_basis_mid_ticks > 0`.",
        "- Scope is not limited to `regime_011_1000_spread_10_20_ticks`.",
        "- Inputs are existing local canonical event-mode public artifacts only.",
        "",
        "## Result",
        "",
        f"- Final recommendation: `{recommendation}`",
        f"- Reason: {reason}",
        f"- Primary row count: `{manifest['basis_positive_primary_row_count']}`",
        f"- Canonical sample count: `{manifest['canonical_sample_count']}`",
        "",
        "## Boundary",
        "",
        "- This report is a read-only public-data proxy context diagnosis, not executable strategy PnL or private execution proof.",
        "- No executable trading instruction, actual order side, quote price, size, leverage, stop rule, take-profit rule, strategy action, case-library trigger, shadow decision, private/order endpoint, order lifecycle, live/default-on/tiny-live, parameter search, deployment recommendation, or promotion is authorized.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_basis_positive_robustness(
    *,
    input_dir: str | Path,
    basis_visibility_dir: str | Path,
    data_contract_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    resolved_input = _expand(input_dir)
    resolved_t005 = _expand(basis_visibility_dir)
    resolved_contract = _expand(data_contract_dir)
    resolved_output = _expand(output_dir)
    guard = canonical_loader.guard_canonical_event_mode_evidence(
        input_dir=resolved_input,
        require_formal_evidence=True,
        allow_diagnostic_validation=False,
    )
    canonical_loader.validate_canonical_source_lock_manifest(
        guard["canonical_source_lock_manifest"],
        require_formal_evidence=True,
    )
    if guard["diagnostic_rejection_count"] != 0:
        raise BasisPositiveRobustnessInputError("T006 refuses diagnostic synthetic formal input")
    t005_manifest = _validate_t005(resolved_t005)
    contract_row = _validate_contract(resolved_contract)
    loaded = guard["loaded_evidence"]
    primary_rows = _primary_rows(_load_horizon_rows(loaded, PRIMARY_HORIZON_MS))
    if not primary_rows:
        raise BasisPositiveRobustnessInputError("no basis-positive primary rows found")
    low_vol, high_vol = _volatility_thresholds(primary_rows)
    overall_row = _summary_row("overall", "context_basis_mid_ticks_gt_0", primary_rows)
    by_sample = _group_summary(primary_rows, "sample", lambda row: row.get("sample_id", ""))
    by_spread = _group_summary(
        primary_rows,
        "spread",
        lambda row: _spread_bucket(_as_float(row.get("context_hyperliquid_spread_ticks"))),
    )
    by_join_age = _group_summary(
        primary_rows,
        "join_age",
        lambda row: _join_age_bucket(_as_float(row.get("context_hyperliquid_join_age_ms"))),
    )
    by_volatility = _group_summary(
        primary_rows,
        "volatility",
        lambda row: _volatility_bucket(_as_float(row.get("input_binance_mid_move_ticks_from_prev")), low_vol, high_vol),
    )
    by_book_state = (
        _group_summary(
            primary_rows,
            "hl_top5_imbalance_sign",
            lambda row: _signed_bucket(_as_float(row.get("context_hyperliquid_top5_imbalance")), "hl_top5_imbalance"),
        )
        + _group_summary(
            primary_rows,
            "hl_microprice_minus_mid_sign",
            lambda row: _signed_bucket(_as_float(row.get("context_hyperliquid_microprice_minus_mid_ticks")), "hl_microprice"),
        )
    )
    collinearity = _collinearity_rows(primary_rows)
    cost_tail = _cost_tail_row(primary_rows)
    horizon_rows = _horizon_persistence_rows(loaded)
    final_recommendation, reason = _final_recommendation(
        overall=overall_row,
        by_spread=by_spread,
        by_join_age=by_join_age,
        by_book=by_book_state,
        collinearity=collinearity,
        cost_tail=cost_tail,
        horizons=horizon_rows,
    )
    if final_recommendation not in FINAL_RECOMMENDATIONS:
        raise AssertionError(f"unexpected recommendation: {final_recommendation}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_dir": str(resolved_input),
        "basis_visibility_dir": str(resolved_t005),
        "data_contract_dir": str(resolved_contract),
        "output_dir": str(resolved_output),
        "canonical_sample_count": guard["canonical_sample_count"],
        "diagnostic_rejection_count": guard["diagnostic_rejection_count"],
        "row_level_input_policy": "multi_sample_manifest.samples[].pricing_signal_rows",
        "assessed_pattern": "context_basis_mid_ticks > 0",
        "scope_policy": "not_limited_to_regime_011",
        "t005_final_contract_decision": t005_manifest.get("final_contract_decision"),
        "contract_basis_mid_decision": contract_row.get("decision"),
        "contract_basis_mid_status": contract_row.get("status"),
        "basis_positive_primary_row_count": len(primary_rows),
        "final_recommendation": final_recommendation,
        "final_recommendation_reason": reason,
        "allowed_final_recommendations": sorted(FINAL_RECOMMENDATIONS),
        "cost_assumptions": {
            "fee_proxy_ticks": FEE_PROXY_TICKS,
            "slippage_proxy_ticks": SLIPPAGE_PROXY_TICKS,
            "latency_decay_proxy_ticks": LATENCY_DECAY_PROXY_TICKS,
            "policy": "fixed_conservative_proxy_not_optimized",
        },
        "output_artifacts": {
            "basis_positive_robustness_manifest": str(resolved_output / "basis_positive_robustness_manifest.json"),
            "basis_positive_overall_summary": str(resolved_output / "basis_positive_overall_summary.csv"),
            "basis_positive_by_sample": str(resolved_output / "basis_positive_by_sample.csv"),
            "basis_positive_by_spread": str(resolved_output / "basis_positive_by_spread.csv"),
            "basis_positive_by_join_age": str(resolved_output / "basis_positive_by_join_age.csv"),
            "basis_positive_by_volatility": str(resolved_output / "basis_positive_by_volatility.csv"),
            "basis_positive_by_hl_book_state": str(resolved_output / "basis_positive_by_hl_book_state.csv"),
            "basis_positive_collinearity": str(resolved_output / "basis_positive_collinearity.csv"),
            "basis_positive_cost_tail": str(resolved_output / "basis_positive_cost_tail.csv"),
            "basis_positive_horizon_persistence": str(resolved_output / "basis_positive_horizon_persistence.csv"),
            "basis_positive_recommendation": str(resolved_output / "basis_positive_recommendation.md"),
        },
        "boundary_flags": BOUNDARY_FLAGS,
    }
    _write_csv(resolved_output / "basis_positive_overall_summary.csv", [overall_row], list(overall_row))
    _write_csv(resolved_output / "basis_positive_by_sample.csv", by_sample, list(by_sample[0]))
    _write_csv(resolved_output / "basis_positive_by_spread.csv", by_spread, list(by_spread[0]))
    _write_csv(resolved_output / "basis_positive_by_join_age.csv", by_join_age, list(by_join_age[0]))
    _write_csv(resolved_output / "basis_positive_by_volatility.csv", by_volatility, list(by_volatility[0]))
    _write_csv(resolved_output / "basis_positive_by_hl_book_state.csv", by_book_state, list(by_book_state[0]))
    _write_csv(resolved_output / "basis_positive_collinearity.csv", collinearity, list(collinearity[0]))
    _write_csv(resolved_output / "basis_positive_cost_tail.csv", [cost_tail], list(cost_tail))
    _write_csv(resolved_output / "basis_positive_horizon_persistence.csv", horizon_rows, list(horizon_rows[0]))
    _write_json(resolved_output / "basis_positive_robustness_manifest.json", manifest)
    _write_report(resolved_output / "basis_positive_recommendation.md", manifest, final_recommendation, reason)
    return {
        "manifest": manifest,
        "overall_rows": [overall_row],
        "sample_rows": by_sample,
        "spread_rows": by_spread,
        "join_age_rows": by_join_age,
        "volatility_rows": by_volatility,
        "book_state_rows": by_book_state,
        "collinearity_rows": collinearity,
        "cost_tail_rows": [cost_tail],
        "horizon_rows": horizon_rows,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--basis-visibility-dir", type=Path, default=DEFAULT_BASIS_VISIBILITY_DIR)
    parser.add_argument("--data-contract-dir", type=Path, default=DEFAULT_DATA_CONTRACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_basis_positive_robustness(
        input_dir=args.input_dir,
        basis_visibility_dir=args.basis_visibility_dir,
        data_contract_dir=args.data_contract_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "final_recommendation": result["manifest"]["final_recommendation"],
                "output_dir": str(_expand(args.output_dir)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
