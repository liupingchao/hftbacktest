#!/usr/bin/env python3
"""Validate maker-optimization preconditions from audit replay reports."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckResult:
    name: str
    path: str
    actual: Any
    expected: Any
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "actual": self.actual,
            "expected": self.expected,
            "passed": self.passed,
            "detail": self.detail,
        }


def _get(data: dict[str, Any], path: str) -> Any:
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(path)
        node = node[part]
    return node


def _get_optional(data: dict[str, Any], path: str, default: Any = None) -> Any:
    try:
        return _get(data, path)
    except KeyError:
        return default


def _check_equal(data: dict[str, Any], path: str, expected: Any, *, name: str = "") -> CheckResult:
    try:
        actual = _get(data, path)
    except KeyError:
        return CheckResult(name or path, path, None, expected, False, "missing field")
    return CheckResult(name or path, path, actual, expected, actual == expected)


def _check_rate_one(
    data: dict[str, Any],
    path: str,
    *,
    tolerance: float,
    name: str = "",
) -> CheckResult:
    try:
        actual = float(_get(data, path))
    except KeyError:
        return CheckResult(name or path, path, None, 1.0, False, "missing field")
    except (TypeError, ValueError):
        return CheckResult(name or path, path, _get(data, path), 1.0, False, "not numeric")
    passed = actual >= 1.0 - float(tolerance)
    return CheckResult(name or path, path, actual, f">= {1.0 - float(tolerance)}", passed)


def _check_abs_diff(
    left: Any,
    right: Any,
    *,
    max_abs_diff: float,
    name: str,
    path: str,
) -> CheckResult:
    try:
        actual = abs(float(left) - float(right))
    except (TypeError, ValueError):
        return CheckResult(name, path, {"left": left, "right": right}, f"<= {max_abs_diff}", False, "not numeric")
    return CheckResult(name, path, actual, f"<= {max_abs_diff}", actual <= float(max_abs_diff))


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _check_bool_true(data: dict[str, Any], path: str, *, name: str = "") -> CheckResult:
    if path not in data:
        return CheckResult(name or path, path, None, True, False, "missing field")
    actual = data[path]
    parsed = _as_bool(actual)
    if parsed is None:
        return CheckResult(name or path, path, actual, True, False, "not boolean-like")
    return CheckResult(name or path, path, actual, True, parsed is True)


def _check_numeric_min(
    data: dict[str, Any],
    path: str,
    minimum: float,
    *,
    name: str = "",
) -> CheckResult:
    if path not in data:
        return CheckResult(name or path, path, None, f">= {minimum}", False, "missing field")
    actual = data[path]
    try:
        value = float(actual)
    except (TypeError, ValueError):
        return CheckResult(name or path, path, actual, f">= {minimum}", False, "not numeric")
    return CheckResult(name or path, path, actual, f">= {minimum}", value >= float(minimum))


def _check_numeric_max(
    data: dict[str, Any],
    path: str,
    maximum: float,
    *,
    name: str = "",
) -> CheckResult:
    if path not in data:
        return CheckResult(name or path, path, None, f"<= {maximum}", False, "missing field")
    actual = data[path]
    try:
        value = float(actual)
    except (TypeError, ValueError):
        return CheckResult(name or path, path, actual, f"<= {maximum}", False, "not numeric")
    return CheckResult(name or path, path, actual, f"<= {maximum}", value <= float(maximum))


def _check_csv_columns(path: Path, required: set[str], *, name: str) -> CheckResult:
    if not path.exists():
        return CheckResult(name, str(path), None, sorted(required), False, "missing file")
    try:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
    except StopIteration:
        return CheckResult(name, str(path), [], sorted(required), False, "empty csv")
    actual = set(header)
    missing = sorted(required - actual)
    return CheckResult(
        name,
        str(path),
        sorted(actual),
        sorted(required),
        not missing,
        "" if not missing else f"missing columns: {', '.join(missing)}",
    )


def _rate(numerator: Any, denominator: Any) -> float | None:
    try:
        den = float(denominator)
        if den <= 0:
            return None
        return float(numerator) / den
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _market_view_classification(
    *,
    action_path_passed: bool,
    required_failures: list[CheckResult],
    quality_failures: list[CheckResult],
) -> str:
    if not action_path_passed:
        return "unusable"
    if required_failures:
        return "compressed_action_path_only"
    if quality_failures:
        return "limited_pricing_research"
    return "passes_pricing_research_market_view"


def evaluate_market_view_acceptance(
    alignment: dict[str, Any],
    *,
    action_path_passed: bool,
    sidecar_metrics: dict[str, Any] | None = None,
    joined_decision_metrics: dict[str, Any] | None = None,
    top5_sidecar_csv: Path | None = None,
    joined_decisions_csv: Path | None = None,
    min_final_data_row_mapping_coverage: float = 1.0,
    min_decision_join_coverage: float = 1.0,
    max_future_join_count: int = 0,
    max_join_missing_count: int = 0,
    max_gap_crossed_join_count: int = 0,
    max_depth_pu_mismatch_count: int = 0,
    max_bookticker_depth_bbo_mismatch_rate: float = 0.001,
    max_stale_join_rate: float = 0.02,
    max_top5_join_age_ms_p99: float = 50.0,
    min_best_bid_tick_match_rate: float = 0.80,
    min_best_ask_tick_match_rate: float = 0.80,
    min_top5_tick_match_rate: float = 0.80,
    min_top5_qty_match_rate: float = 0.75,
) -> dict[str, Any]:
    """Evaluate optional Stage 3 market-view quality gates."""

    enabled = sidecar_metrics is not None or joined_decision_metrics is not None
    top5_book_state = alignment.get("top5_book_state", {})
    if not enabled:
        return {
            "enabled": False,
            "passed": None,
            "classification": "not_evaluated",
            "hard_failures": [],
            "checks": [],
            "diagnostics": {"top5_book_state": top5_book_state},
        }

    sidecar = sidecar_metrics or {}
    joined = joined_decision_metrics or {}
    required_checks: list[CheckResult] = []
    quality_checks: list[CheckResult] = []

    if sidecar_metrics is None:
        required_checks.append(
            CheckResult(
                "sidecar metrics present",
                "sidecar_metrics",
                None,
                "metrics json",
                False,
                "missing sidecar metrics",
            )
        )
    if joined_decision_metrics is None:
        required_checks.append(
            CheckResult(
                "joined decision metrics present",
                "joined_decision_metrics",
                None,
                "metrics json",
                False,
                "missing joined decision metrics",
            )
        )

    required_checks.extend(
        [
            _check_bool_true(sidecar, "first_valid_update_aligned"),
            _check_numeric_max(sidecar, "depth_pu_mismatch_count", max_depth_pu_mismatch_count),
            _check_numeric_min(
                sidecar,
                "final_data_row_mapping_coverage",
                min_final_data_row_mapping_coverage,
            ),
            _check_numeric_min(joined, "decision_join_coverage", min_decision_join_coverage),
            _check_numeric_max(joined, "future_join_count", max_future_join_count),
            _check_numeric_max(joined, "join_missing_count", max_join_missing_count),
            _check_numeric_max(joined, "gap_crossed_join_count", max_gap_crossed_join_count),
        ]
    )

    if top5_sidecar_csv is not None:
        required_checks.append(
            _check_csv_columns(
                top5_sidecar_csv,
                {
                    "raw_seq",
                    "event_type",
                    "local_ts",
                    "depth_U",
                    "depth_u",
                    "pu",
                    "snapshot_lastUpdateId",
                    "sync_waiting_snapshot",
                    "sync_aligned",
                    "sync_gap",
                    "startup_excluded",
                    "first_valid_update_aligned",
                    "bid_top5_ticks",
                    "bid_top5_qtys",
                    "ask_top5_ticks",
                    "ask_top5_qtys",
                    "bookticker_bbo_match",
                    "bookticker_depth_age_ms",
                },
                name="top5 sidecar provenance columns",
            )
        )
    if joined_decisions_csv is not None:
        required_checks.append(
            _check_csv_columns(
                joined_decisions_csv,
                {
                    "strategy_seq",
                    "join_key",
                    "join_used_future",
                    "join_missing",
                    "join_stale",
                    "join_gap_crossed",
                    "joined_raw_seq",
                    "joined_depth_u",
                    "top5_join_age_ms",
                    "depth_join_age_ms",
                    "bookticker_join_age_ms",
                    "max_join_age_ms",
                    "joined_top5_source",
                },
                name="joined decision provenance columns",
            )
        )

    bbo_match_count = sidecar.get("bookticker_depth_bbo_match_count")
    bbo_mismatch_count = sidecar.get("bookticker_depth_bbo_mismatch_count")
    bbo_mismatch_rate = _rate(bbo_mismatch_count, float(bbo_match_count or 0) + float(bbo_mismatch_count or 0))
    quality_checks.append(
        CheckResult(
            "bookTicker/depth BBO mismatch rate",
            "bookticker_depth_bbo_mismatch_count / bookticker_depth_bbo_total",
            bbo_mismatch_rate,
            f"<= {max_bookticker_depth_bbo_mismatch_rate}",
            bbo_mismatch_rate is not None and bbo_mismatch_rate <= float(max_bookticker_depth_bbo_mismatch_rate),
            "" if bbo_mismatch_rate is not None else "missing BBO match/mismatch counts",
        )
    )

    stale_join_count = joined.get("stale_join_count")
    decision_count = joined.get("decision_count")
    stale_join_rate = _rate(stale_join_count, decision_count)
    quality_checks.append(
        CheckResult(
            "stale join rate",
            "stale_join_count / decision_count",
            stale_join_rate,
            f"<= {max_stale_join_rate}",
            stale_join_rate is not None and stale_join_rate <= float(max_stale_join_rate),
            "" if stale_join_rate is not None else "missing stale_join_count or decision_count",
        )
    )
    quality_checks.append(
        _check_numeric_max(joined, "top5_join_age_ms_p99", max_top5_join_age_ms_p99)
    )

    quality_checks.extend(
        [
            _check_numeric_min(
                top5_book_state,
                "bid_tick_match_rate",
                min_best_bid_tick_match_rate,
                name="best bid tick match rate",
            ),
            _check_numeric_min(
                top5_book_state,
                "ask_tick_match_rate",
                min_best_ask_tick_match_rate,
                name="best ask tick match rate",
            ),
            _check_numeric_min(
                top5_book_state,
                "top5_tick_match_rate",
                min_top5_tick_match_rate,
            ),
            _check_numeric_min(
                top5_book_state,
                "top5_qty_match_rate",
                min_top5_qty_match_rate,
            ),
        ]
    )

    required_failures = [check for check in required_checks if not check.passed]
    quality_failures = [check for check in quality_checks if not check.passed]
    classification = _market_view_classification(
        action_path_passed=action_path_passed,
        required_failures=required_failures,
        quality_failures=quality_failures,
    )
    passed = classification == "passes_pricing_research_market_view"
    return {
        "enabled": True,
        "passed": passed,
        "classification": classification,
        "hard_failures": [check.to_dict() for check in required_failures + quality_failures],
        "checks": [check.to_dict() for check in required_checks + quality_checks],
        "diagnostics": {
            "thresholds": {
                "min_final_data_row_mapping_coverage": min_final_data_row_mapping_coverage,
                "min_decision_join_coverage": min_decision_join_coverage,
                "max_future_join_count": max_future_join_count,
                "max_join_missing_count": max_join_missing_count,
                "max_gap_crossed_join_count": max_gap_crossed_join_count,
                "max_depth_pu_mismatch_count": max_depth_pu_mismatch_count,
                "max_bookticker_depth_bbo_mismatch_rate": max_bookticker_depth_bbo_mismatch_rate,
                "max_stale_join_rate": max_stale_join_rate,
                "max_top5_join_age_ms_p99": max_top5_join_age_ms_p99,
                "min_best_bid_tick_match_rate": min_best_bid_tick_match_rate,
                "min_best_ask_tick_match_rate": min_best_ask_tick_match_rate,
                "min_top5_tick_match_rate": min_top5_tick_match_rate,
                "min_top5_qty_match_rate": min_top5_qty_match_rate,
            },
            "sidecar_metrics": sidecar,
            "joined_decision_metrics": joined,
            "derived": {
                "bookticker_depth_bbo_mismatch_rate": bbo_mismatch_rate,
                "stale_join_rate": stale_join_rate,
            },
            "top5_book_state": top5_book_state,
            "scope_note": (
                "Top5 tick/qty match rates are market-view quality thresholds, "
                "not full L2 or exact queue-position proof."
            ),
        },
    }


def evaluate_maker_acceptance(
    alignment_report: dict[str, Any],
    *,
    backtest_result: dict[str, Any] | None = None,
    sidecar_metrics: dict[str, Any] | None = None,
    joined_decision_metrics: dict[str, Any] | None = None,
    top5_sidecar_csv: Path | None = None,
    joined_decisions_csv: Path | None = None,
    alignment_key: str = "alignment",
    min_common_rows: int = 1,
    rate_tolerance: float = 0.0,
    max_latency_drop_abs_diff: float = 1e-4,
    max_api_drop_abs_diff: float = 1e-4,
    min_final_data_row_mapping_coverage: float = 1.0,
    min_decision_join_coverage: float = 1.0,
    max_future_join_count: int = 0,
    max_join_missing_count: int = 0,
    max_gap_crossed_join_count: int = 0,
    max_depth_pu_mismatch_count: int = 0,
    max_bookticker_depth_bbo_mismatch_rate: float = 0.001,
    max_stale_join_rate: float = 0.02,
    max_top5_join_age_ms_p99: float = 50.0,
    min_best_bid_tick_match_rate: float = 0.80,
    min_best_ask_tick_match_rate: float = 0.80,
    min_top5_tick_match_rate: float = 0.80,
    min_top5_qty_match_rate: float = 0.75,
) -> dict[str, Any]:
    """Return a structured maker-optimization acceptance result."""

    alignment = _get(alignment_report, alignment_key)
    checks: list[CheckResult] = [
        _check_rate_one(alignment, "action_match_rate", tolerance=rate_tolerance),
        _check_rate_one(alignment, "planned_action_match_rate", tolerance=rate_tolerance),
        _check_rate_one(alignment, "reject_reason_match_rate", tolerance=rate_tolerance),
        _check_rate_one(alignment, "throttle_reason_match_rate", tolerance=rate_tolerance),
        _check_equal(alignment, "working_order_lifecycle.semantic_mismatch_rows", 0),
        _check_equal(alignment, "working_order_lifecycle.blocking_mismatch_rows", 0),
        _check_equal(alignment, "api_throttle.mismatch_attribution.mismatch_rows", 0),
        _check_equal(alignment, "api_throttle.mismatch_attribution.target_tick_mismatch_rows", 0),
        _check_equal(alignment, "replay_lag.missing_lag_rows", 0),
        _check_equal(alignment, "replay_lag.missing_exchange_lag_rows", 0),
        _check_equal(alignment, "replay_lag.stateful_gate.startup_excluded_gate.passed", True),
        _check_equal(
            alignment,
            "replay_lag.stateful_gate.startup_excluded_gate.post_startup_outside_dual_gate_rows",
            0,
        ),
    ]

    common_rows = _get_optional(alignment, "common_rows")
    try:
        common_rows_passed = int(common_rows) >= int(min_common_rows)
        minimum_detail = ""
    except (TypeError, ValueError):
        common_rows_passed = False
        minimum_detail = "missing or not integer"
    checks.append(
        CheckResult(
            "minimum common rows",
            "common_rows",
            common_rows,
            f">= {int(min_common_rows)}",
            common_rows_passed,
            minimum_detail,
        )
    )

    bt_summary = alignment_report.get("bt_summary", {})
    live_summary = alignment_report.get("live_summary", {})
    checks.append(
        _check_abs_diff(
            bt_summary.get("drop_latency_rate"),
            live_summary.get("drop_latency_rate"),
            max_abs_diff=max_latency_drop_abs_diff,
            name="BT/live latency drop aligned",
            path="bt_summary.drop_latency_rate vs live_summary.drop_latency_rate",
        )
    )
    checks.append(
        _check_abs_diff(
            bt_summary.get("drop_api_rate"),
            live_summary.get("drop_api_rate"),
            max_abs_diff=max_api_drop_abs_diff,
            name="BT/live API drop aligned",
            path="bt_summary.drop_api_rate vs live_summary.drop_api_rate",
        )
    )

    if backtest_result is not None:
        gate_path = "audit_replay_lag_gate"
        checks.extend(
            [
                _check_equal(backtest_result, f"{gate_path}.enabled", True, name="strict replay gate enabled"),
                _check_equal(backtest_result, f"{gate_path}.strict", True, name="strict replay gate strict"),
                _check_equal(backtest_result, f"{gate_path}.passed", True, name="strict replay gate passed"),
                _check_equal(backtest_result, f"{gate_path}.breach_count", 0, name="strict replay gate breaches"),
                _check_equal(backtest_result, f"{gate_path}.drop_count", 0, name="strict replay gate drops"),
                _check_equal(backtest_result, f"{gate_path}.fail_count", 0, name="strict replay gate failures"),
            ]
        )

    diagnostics = {
        "working_order_non_blocking_mismatch_rows": _get_optional(
            alignment,
            "working_order_lifecycle.non_blocking_mismatch_rows",
        ),
        "working_order_identity_only_mismatch_rows": _get_optional(
            alignment,
            "working_order_lifecycle.identity_only_mismatch_rows",
        ),
        "working_order_diagnostic_mismatch_rows": _get_optional(
            alignment,
            "working_order_lifecycle.diagnostic_mismatch_rows",
        ),
        "working_order_rest_local_divergence_rows": _get_optional(
            alignment,
            "working_order_lifecycle.rest_local_divergence_rows",
        ),
        "top5_book_state": alignment.get("top5_book_state", {}),
    }

    base_failed = [check for check in checks if not check.passed]
    market_view = evaluate_market_view_acceptance(
        alignment,
        action_path_passed=not base_failed,
        sidecar_metrics=sidecar_metrics,
        joined_decision_metrics=joined_decision_metrics,
        top5_sidecar_csv=top5_sidecar_csv,
        joined_decisions_csv=joined_decisions_csv,
        min_final_data_row_mapping_coverage=min_final_data_row_mapping_coverage,
        min_decision_join_coverage=min_decision_join_coverage,
        max_future_join_count=max_future_join_count,
        max_join_missing_count=max_join_missing_count,
        max_gap_crossed_join_count=max_gap_crossed_join_count,
        max_depth_pu_mismatch_count=max_depth_pu_mismatch_count,
        max_bookticker_depth_bbo_mismatch_rate=max_bookticker_depth_bbo_mismatch_rate,
        max_stale_join_rate=max_stale_join_rate,
        max_top5_join_age_ms_p99=max_top5_join_age_ms_p99,
        min_best_bid_tick_match_rate=min_best_bid_tick_match_rate,
        min_best_ask_tick_match_rate=min_best_ask_tick_match_rate,
        min_top5_tick_match_rate=min_top5_tick_match_rate,
        min_top5_qty_match_rate=min_top5_qty_match_rate,
    )
    market_failed = [
        CheckResult(
            item["name"],
            item["path"],
            item["actual"],
            item["expected"],
            False,
            item.get("detail", ""),
        )
        for item in market_view["hard_failures"]
    ]
    failed = base_failed + market_failed
    return {
        "passed": not failed,
        "alignment_key": alignment_key,
        "hard_failures": [check.to_dict() for check in failed],
        "checks": [check.to_dict() for check in checks],
        "diagnostics": diagnostics,
        "market_view": market_view,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate maker optimization acceptance gates")
    parser.add_argument("--alignment-report", required=True, help="Path to alignment_report_audit_replay.json")
    parser.add_argument("--backtest-result", default="", help="Optional backtest_audit_replay_result.json")
    parser.add_argument("--sidecar-metrics", default="", help="Optional T009 sidecar metrics.json")
    parser.add_argument(
        "--joined-decision-metrics",
        default="",
        help="Optional T009 joined_decisions.metrics.json",
    )
    parser.add_argument("--top5-sidecar-csv", default="", help="Optional top5_sidecar.csv for provenance columns")
    parser.add_argument(
        "--joined-decisions-csv",
        default="",
        help="Optional joined_decisions.csv for provenance columns",
    )
    parser.add_argument("--alignment-key", default="alignment", help="Alignment report key to validate")
    parser.add_argument("--min-common-rows", type=int, default=1)
    parser.add_argument("--rate-tolerance", type=float, default=0.0)
    parser.add_argument("--max-latency-drop-abs-diff", type=float, default=1e-4)
    parser.add_argument("--max-api-drop-abs-diff", type=float, default=1e-4)
    parser.add_argument("--min-final-data-row-mapping-coverage", type=float, default=1.0)
    parser.add_argument("--min-decision-join-coverage", type=float, default=1.0)
    parser.add_argument("--max-future-join-count", type=int, default=0)
    parser.add_argument("--max-join-missing-count", type=int, default=0)
    parser.add_argument("--max-gap-crossed-join-count", type=int, default=0)
    parser.add_argument("--max-depth-pu-mismatch-count", type=int, default=0)
    parser.add_argument("--max-bookticker-depth-bbo-mismatch-rate", type=float, default=0.001)
    parser.add_argument("--max-stale-join-rate", type=float, default=0.02)
    parser.add_argument("--max-top5-join-age-ms-p99", type=float, default=50.0)
    parser.add_argument("--min-best-bid-tick-match-rate", type=float, default=0.80)
    parser.add_argument("--min-best-ask-tick-match-rate", type=float, default=0.80)
    parser.add_argument("--min-top5-tick-match-rate", type=float, default=0.80)
    parser.add_argument("--min-top5-qty-match-rate", type=float, default=0.75)
    parser.add_argument("--out", default="", help="Optional output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alignment_report = json.loads(Path(args.alignment_report).read_text())
    backtest_result = (
        json.loads(Path(args.backtest_result).read_text()) if args.backtest_result else None
    )
    sidecar_metrics = json.loads(Path(args.sidecar_metrics).read_text()) if args.sidecar_metrics else None
    joined_decision_metrics = (
        json.loads(Path(args.joined_decision_metrics).read_text())
        if args.joined_decision_metrics
        else None
    )
    result = evaluate_maker_acceptance(
        alignment_report,
        backtest_result=backtest_result,
        sidecar_metrics=sidecar_metrics,
        joined_decision_metrics=joined_decision_metrics,
        top5_sidecar_csv=Path(args.top5_sidecar_csv) if args.top5_sidecar_csv else None,
        joined_decisions_csv=Path(args.joined_decisions_csv) if args.joined_decisions_csv else None,
        alignment_key=str(args.alignment_key),
        min_common_rows=int(args.min_common_rows),
        rate_tolerance=float(args.rate_tolerance),
        max_latency_drop_abs_diff=float(args.max_latency_drop_abs_diff),
        max_api_drop_abs_diff=float(args.max_api_drop_abs_diff),
        min_final_data_row_mapping_coverage=float(args.min_final_data_row_mapping_coverage),
        min_decision_join_coverage=float(args.min_decision_join_coverage),
        max_future_join_count=int(args.max_future_join_count),
        max_join_missing_count=int(args.max_join_missing_count),
        max_gap_crossed_join_count=int(args.max_gap_crossed_join_count),
        max_depth_pu_mismatch_count=int(args.max_depth_pu_mismatch_count),
        max_bookticker_depth_bbo_mismatch_rate=float(args.max_bookticker_depth_bbo_mismatch_rate),
        max_stale_join_rate=float(args.max_stale_join_rate),
        max_top5_join_age_ms_p99=float(args.max_top5_join_age_ms_p99),
        min_best_bid_tick_match_rate=float(args.min_best_bid_tick_match_rate),
        min_best_ask_tick_match_rate=float(args.min_best_ask_tick_match_rate),
        min_top5_tick_match_rate=float(args.min_top5_tick_match_rate),
        min_top5_qty_match_rate=float(args.min_top5_qty_match_rate),
    )

    payload = json.dumps(result, indent=2, ensure_ascii=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n")
    print(payload)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
