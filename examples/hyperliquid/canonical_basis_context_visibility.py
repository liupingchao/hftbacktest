#!/usr/bin/env python3
"""Read-only basis context visibility and lineage diagnosis for Regime 011."""

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
TASK_ID = "0608T005"
SCHEMA_VERSION = "canonical_basis_context_visibility_v1"
TARGET_REGIME_ID = momentum.TARGET_REGIME_ID
TARGET_HORIZON_MS = 1000
TICK_SIZE = 0.1

DEFAULT_INPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "event_mode_canonical_pricing_signal_0604T003"
DEFAULT_FEATURE_VALIDITY_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_feature_conditioned_signal_validity_0608T004"
DEFAULT_DATA_CONTRACT_DIR = PROJECT_ROOT / "local_live_analysis" / "binance_led_hyperliquid_data_contract_0601T004"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_basis_context_visibility_0608T005"

BOUNDARY_FLAGS = {
    "no_new_data_collection": True,
    "read_only_basis_context_visibility_lineage_diagnosis_only": True,
    "no_executable_trading_instruction": True,
    "no_actual_order_side_output": True,
    "no_quote_price_or_size_output": True,
    "no_leverage_output": True,
    "no_stop_or_take_profit_rule": True,
    "no_deployment_recommendation": True,
    "no_case_library_implementation": True,
    "no_shadow_decision_generation": True,
    "no_private_account_endpoints": True,
    "no_order_endpoints": True,
    "no_order_lifecycle": True,
    "no_strategy_implementation": True,
    "no_live_trading_bot": True,
    "no_parameter_search": True,
    "no_default_on": True,
    "no_tiny_live": True,
    "no_promotion": True,
}


class BasisVisibilityInputError(ValueError):
    """Raised when T005 inputs violate the read-only diagnosis contract."""


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
        raise BasisVisibilityInputError(f"{path} must contain a JSON object")
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


def _float(value: Any) -> float | None:
    return momentum._as_float(value)


def _int(value: Any, default: int = 0) -> int:
    return momentum._as_int(value, default=default)


def _fmt(value: float | None) -> str:
    return momentum._format_float(value)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _validate_t004(feature_validity_dir: Path) -> dict[str, Any]:
    manifest = _read_json(feature_validity_dir / "feature_validity_manifest.json")
    if manifest.get("task_id") != "0608T004":
        raise BasisVisibilityInputError("feature validity manifest must be from 0608T004")
    if manifest.get("schema_version") != "canonical_feature_conditioned_signal_validity_v1":
        raise BasisVisibilityInputError("unexpected T004 schema")
    if manifest.get("assessed_regime_id") != TARGET_REGIME_ID:
        raise BasisVisibilityInputError("T004 regime does not match T005 scope")
    if manifest.get("final_recommendation") != "watch_needs_contract_visibility_clarification":
        raise BasisVisibilityInputError("T005 requires T004 contract-visibility watch result")
    rows = _read_csv(feature_validity_dir / "feature_pattern_validity_summary.csv")
    matches = [
        row
        for row in rows
        if row.get("feature") == "context_basis_mid_ticks" and row.get("variant") == "sign_positive"
    ]
    if len(matches) != 1:
        raise BasisVisibilityInputError("expected exactly one T004 basis positive row")
    row = matches[0]
    if row.get("signal_validity") != "invalid_not_decision_visible":
        raise BasisVisibilityInputError("T004 basis positive row is not blocked only by visibility")
    if row.get("stability_classification") != "stable_all_samples":
        raise BasisVisibilityInputError("T004 basis positive row is not stable across samples")
    if row.get("tail_risk_classification") != "tail_risk_acceptable_proxy":
        raise BasisVisibilityInputError("T004 basis positive row did not pass tail proxy")
    return {"manifest": manifest, "basis_positive": row}


def _contract_basis_row(data_contract_dir: Path) -> dict[str, str]:
    rows = _read_csv(data_contract_dir / "feature_decision_table.csv")
    matches = [row for row in rows if row.get("feature") == "basis_mid_dislocation"]
    if len(matches) != 1:
        raise BasisVisibilityInputError("0601T004 contract must contain exactly one basis_mid_dislocation row")
    return matches[0]


def _source_join_rows(sample: dict[str, Any]) -> dict[int, dict[str, str]]:
    join_dir = Path(str(sample.get("source_join_dir", "")))
    rows = _read_csv(join_dir / "cross_exchange_joined_features.csv")
    by_seq: dict[int, dict[str, str]] = {}
    for row in rows:
        by_seq[_int(row.get("join_seq"))] = row
    return by_seq


def _load_rows(loaded: dict[str, Any], horizon_ms: int) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows, _ = momentum._context_rows_from_manifest(loaded, horizon_ms=horizon_ms)
    samples = {
        sample["sample_id"]: sample
        for sample in loaded["source_manifest"].get("samples", [])
        if isinstance(sample, dict)
    }
    return rows, samples


def _basis_positive(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if (_float(row.get("context_basis_mid_ticks")) or 0.0) > 0]


def _lineage_and_timestamp_rows(rows: list[dict[str, str]], samples: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_cache: dict[str, dict[int, dict[str, str]]] = {}
    lineage_rows: list[dict[str, Any]] = []
    timestamp_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row.get("sample_id", "")
        sample = samples[sample_id]
        if sample_id not in source_cache:
            source_cache[sample_id] = _source_join_rows(sample)
        source_row_index = _int(row.get("source_row_index"))
        source = source_cache[sample_id].get(source_row_index, {})
        binance_mid = _float(source.get("binance_mid_px"))
        hl_mid = _float(source.get("hyperliquid_mid_px"))
        joined_basis = _float(source.get("basis_mid_ticks"))
        context_basis = _float(row.get("context_basis_mid_ticks"))
        recomputed = None if binance_mid is None or hl_mid is None else (binance_mid - hl_mid) / TICK_SIZE
        formula_abs_error = None if recomputed is None or context_basis is None else abs(recomputed - context_basis)
        joined_abs_error = None if joined_basis is None or context_basis is None else abs(joined_basis - context_basis)
        lineage_ok = (
            formula_abs_error is not None
            and formula_abs_error <= 1e-6
            and joined_abs_error is not None
            and joined_abs_error <= 1e-6
        )
        decision_ts = _int(row.get("hyperliquid_decision_ts"))
        binance_ts = _int(row.get("binance_local_ts"))
        source_age = _float(row.get("binance_source_age_ms"))
        future_join = binance_ts > decision_ts if binance_ts and decision_ts else True
        missing = not bool(source.get("binance_source_found", "").lower() == "true")
        timestamp_ok = (
            not future_join
            and not missing
            and row.get("joined_row_quality") == "primary_usable"
            and row.get("context_hyperliquid_context_quality") == "primary_usable"
            and row.get("context_hyperliquid_join_age_bucket") == "fresh_0_50ms"
            and source_age is not None
            and source_age <= 50
        )
        lineage_rows.append(
            {
                "sample_id": sample_id,
                "source_row_index": source_row_index,
                "binance_mid_px": source.get("binance_mid_px", ""),
                "hyperliquid_mid_px": source.get("hyperliquid_mid_px", ""),
                "context_basis_mid_ticks": _fmt(context_basis),
                "source_basis_mid_ticks": _fmt(joined_basis),
                "recomputed_basis_mid_ticks": _fmt(recomputed),
                "formula_abs_error": _fmt(formula_abs_error),
                "source_context_abs_error": _fmt(joined_abs_error),
                "lineage_status": "lineage_confirmed_decision_time_formula" if lineage_ok else "lineage_failed",
                "basis_contract_caveat": row.get("basis_contract_caveat", ""),
            }
        )
        timestamp_rows.append(
            {
                "sample_id": sample_id,
                "source_row_index": source_row_index,
                "hyperliquid_decision_ts": row.get("hyperliquid_decision_ts", ""),
                "binance_local_ts": row.get("binance_local_ts", ""),
                "binance_source_age_ms": row.get("binance_source_age_ms", ""),
                "joined_row_quality": row.get("joined_row_quality", ""),
                "hyperliquid_context_quality": row.get("context_hyperliquid_context_quality", ""),
                "hyperliquid_join_age_bucket": row.get("context_hyperliquid_join_age_bucket", ""),
                "future_input_join": int(future_join),
                "missing_input_join": int(missing),
                "timestamp_status": "timestamp_clean_asof" if timestamp_ok else "timestamp_failed_future_join_or_missing",
            }
        )
    return lineage_rows, timestamp_rows


def _sample_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_sample: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _float(row.get("hyperliquid_future_mid_move_ticks"))
        if value is not None:
            by_sample[row.get("sample_id", "")].append(value)
    total = sum(len(values) for values in by_sample.values())
    out: list[dict[str, Any]] = []
    for sample_id, values in sorted(by_sample.items()):
        positive = sum(1 for value in values if value > 0)
        negative = sum(1 for value in values if value < 0)
        out.append(
            {
                "sample_id": sample_id,
                "row_count": len(values),
                "row_share": _fmt(len(values) / total if total else None),
                "positive_future_count": positive,
                "negative_future_count": negative,
                "direction_hit_rate": _fmt(positive / (positive + negative) if positive + negative else None),
                "mean_future_mid_move_ticks": _fmt(_mean(values)),
            }
        )
    return out


def _horizon_persistence(loaded: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for horizon in [100, 250, 1000, 5000, 10000]:
        rows, _ = _load_rows(loaded, horizon)
        selected = _basis_positive(rows)
        moves = [value for value in (_float(row.get("hyperliquid_future_mid_move_ticks")) for row in selected) if value is not None]
        positive = sum(1 for value in moves if value > 0)
        negative = sum(1 for value in moves if value < 0)
        out.append(
            {
                "horizon_ms": horizon,
                "evidence_role": "primary" if horizon == 1000 else ("watch_alias_context" if horizon in {100, 250} else "diagnostic_persistence"),
                "row_count": len(moves),
                "sample_count": len({row.get("sample_id", "") for row in selected}),
                "positive_future_count": positive,
                "negative_future_count": negative,
                "direction_hit_rate": _fmt(positive / (positive + negative) if positive + negative else None),
                "mean_future_mid_move_ticks": _fmt(_mean(moves)),
                "direction": "positive" if (_mean(moves) or 0.0) > 0 else "non_positive",
            }
        )
    return out


def _contract_decision(
    *,
    lineage_rows: list[dict[str, Any]],
    timestamp_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    horizon_rows: list[dict[str, Any]],
    contract_basis: dict[str, str],
) -> dict[str, Any]:
    lineage_ok = all(row["lineage_status"] == "lineage_confirmed_decision_time_formula" for row in lineage_rows)
    timestamp_ok = all(row["timestamp_status"] == "timestamp_clean_asof" for row in timestamp_rows)
    sample_count = len(sample_rows)
    max_share = max((_float(row["row_share"]) or 0.0 for row in sample_rows), default=1.0)
    persistence_ok = all(
        (_float(row["mean_future_mid_move_ticks"]) or 0.0) > 0
        for row in horizon_rows
        if row["horizon_ms"] in {1000, 5000, 10000}
    )
    current_contract = f"{contract_basis.get('decision', '')}/{contract_basis.get('status', '')}"
    if lineage_ok and timestamp_ok and sample_count == 3 and max_share <= 0.55 and persistence_ok:
        decision = "upgrade_to_context_only_supported"
        reason = "basis is as-of clean and formula-derived; retain execution-PnL caveat but remove decision-visibility blocker"
    elif lineage_ok and timestamp_ok:
        decision = "watch_contract_visibility_needs_more_samples"
        reason = "basis lineage is clean but sample concentration or persistence remains watch-only"
    elif not lineage_ok:
        decision = "reject_basis_context"
        reason = "basis formula lineage could not be confirmed"
    else:
        decision = "keep_diagnostic_only_contract_caveat"
        reason = "basis timestamp/as-of cleanliness failed"
    return {
        "regime_id": TARGET_REGIME_ID,
        "pattern": "context_basis_mid_ticks > 0",
        "current_contract_decision": current_contract,
        "lineage_classification": "lineage_confirmed_decision_time_formula" if lineage_ok else "lineage_failed",
        "timestamp_classification": "timestamp_clean_asof" if timestamp_ok else "timestamp_failed_future_join_or_missing",
        "sample_count": sample_count,
        "max_sample_row_share": _fmt(max_share),
        "persistence_1000_5000_10000_positive": int(persistence_ok),
        "contract_decision": decision,
        "decision_reason": reason,
        "scope": "read_only_context_visibility_not_strategy_or_execution_pnl",
    }


def _write_report(path: Path, manifest: dict[str, Any], decision: dict[str, Any]) -> None:
    lines = [
        "# Basis Context Visibility Report",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "## Result",
        "",
        f"- Contract decision: `{decision['contract_decision']}`",
        f"- Lineage: `{decision['lineage_classification']}`",
        f"- Timestamp: `{decision['timestamp_classification']}`",
        f"- Max sample row share: `{decision['max_sample_row_share']}`",
        f"- Decision reason: {decision['decision_reason']}",
        "",
        "## Scope",
        "",
        f"- Assessed regime: `{TARGET_REGIME_ID}` only.",
        "- Assessed pattern: `context_basis_mid_ticks > 0` only.",
        "- Basis formula audited as `(binance_mid_px - hyperliquid_mid_px) / 0.1` from decision-time as-of joined rows.",
        "",
        "## Boundary",
        "",
        "- This report is a read-only public-data proxy visibility / lineage diagnosis, not executable strategy PnL or private execution proof.",
        "- No executable trading instruction, actual order side, quote price, size, leverage, stop rule, take-profit rule, strategy action, shadow decision, private/order endpoint, order lifecycle, live/default-on/tiny-live, parameter search, case-library implementation, deployment recommendation, or promotion is authorized.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_basis_context_visibility(
    *,
    input_dir: str | Path,
    feature_validity_dir: str | Path,
    data_contract_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    resolved_input = _expand(input_dir)
    resolved_t004 = _expand(feature_validity_dir)
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
        raise BasisVisibilityInputError("T005 refuses diagnostic synthetic formal input")
    t004 = _validate_t004(resolved_t004)
    contract_basis = _contract_basis_row(resolved_contract)
    loaded = guard["loaded_evidence"]
    rows_1000, samples = _load_rows(loaded, TARGET_HORIZON_MS)
    positive = _basis_positive(rows_1000)
    if not positive:
        raise BasisVisibilityInputError("no basis-positive target rows found")
    lineage_rows, timestamp_rows = _lineage_and_timestamp_rows(positive, samples)
    sample_rows = _sample_summary(positive)
    horizon_rows = _horizon_persistence(loaded)
    decision = _contract_decision(
        lineage_rows=lineage_rows,
        timestamp_rows=timestamp_rows,
        sample_rows=sample_rows,
        horizon_rows=horizon_rows,
        contract_basis=contract_basis,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_dir": str(resolved_input),
        "feature_validity_dir": str(resolved_t004),
        "data_contract_dir": str(resolved_contract),
        "output_dir": str(resolved_output),
        "canonical_sample_count": guard["canonical_sample_count"],
        "diagnostic_rejection_count": guard["diagnostic_rejection_count"],
        "assessed_regime_id": TARGET_REGIME_ID,
        "assessed_pattern": "context_basis_mid_ticks > 0",
        "row_level_input_policy": "multi_sample_manifest.samples[].pricing_signal_rows",
        "basis_formula": "(binance_mid_px - hyperliquid_mid_px) / 0.1",
        "t004_basis_positive_signal_validity": t004["basis_positive"].get("signal_validity"),
        "basis_positive_row_count": len(positive),
        "final_contract_decision": decision["contract_decision"],
        "output_artifacts": {
            "basis_visibility_manifest": str(resolved_output / "basis_visibility_manifest.json"),
            "basis_lineage_audit": str(resolved_output / "basis_lineage_audit.csv"),
            "basis_positive_pattern_by_sample": str(resolved_output / "basis_positive_pattern_by_sample.csv"),
            "basis_timestamp_join_audit": str(resolved_output / "basis_timestamp_join_audit.csv"),
            "basis_horizon_persistence": str(resolved_output / "basis_horizon_persistence.csv"),
            "basis_contract_decision": str(resolved_output / "basis_contract_decision.csv"),
            "basis_context_visibility_report": str(resolved_output / "basis_context_visibility_report.md"),
        },
        "boundary_flags": BOUNDARY_FLAGS,
    }
    _write_csv(resolved_output / "basis_lineage_audit.csv", lineage_rows, list(lineage_rows[0]))
    _write_csv(resolved_output / "basis_timestamp_join_audit.csv", timestamp_rows, list(timestamp_rows[0]))
    _write_csv(resolved_output / "basis_positive_pattern_by_sample.csv", sample_rows, list(sample_rows[0]))
    _write_csv(resolved_output / "basis_horizon_persistence.csv", horizon_rows, list(horizon_rows[0]))
    _write_csv(resolved_output / "basis_contract_decision.csv", [decision], list(decision))
    _write_json(resolved_output / "basis_visibility_manifest.json", manifest)
    _write_report(resolved_output / "basis_context_visibility_report.md", manifest, decision)
    return {
        "manifest": manifest,
        "lineage_rows": lineage_rows,
        "timestamp_rows": timestamp_rows,
        "sample_rows": sample_rows,
        "horizon_rows": horizon_rows,
        "decision_rows": [decision],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--feature-validity-dir", type=Path, default=DEFAULT_FEATURE_VALIDITY_DIR)
    parser.add_argument("--data-contract-dir", type=Path, default=DEFAULT_DATA_CONTRACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_basis_context_visibility(
        input_dir=args.input_dir,
        feature_validity_dir=args.feature_validity_dir,
        data_contract_dir=args.data_contract_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "assessed_regime_id": TARGET_REGIME_ID,
                "final_contract_decision": result["manifest"]["final_contract_decision"],
                "output_dir": str(_expand(args.output_dir)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
