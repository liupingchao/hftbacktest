#!/usr/bin/env python3
"""Validate maker-optimization preconditions from audit replay reports."""

from __future__ import annotations

import argparse
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


def evaluate_maker_acceptance(
    alignment_report: dict[str, Any],
    *,
    backtest_result: dict[str, Any] | None = None,
    alignment_key: str = "alignment",
    min_common_rows: int = 1,
    rate_tolerance: float = 0.0,
    max_latency_drop_abs_diff: float = 1e-4,
    max_api_drop_abs_diff: float = 1e-4,
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

    failed = [check for check in checks if not check.passed]
    return {
        "passed": not failed,
        "alignment_key": alignment_key,
        "hard_failures": [check.to_dict() for check in failed],
        "checks": [check.to_dict() for check in checks],
        "diagnostics": diagnostics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate maker optimization acceptance gates")
    parser.add_argument("--alignment-report", required=True, help="Path to alignment_report_audit_replay.json")
    parser.add_argument("--backtest-result", default="", help="Optional backtest_audit_replay_result.json")
    parser.add_argument("--alignment-key", default="alignment", help="Alignment report key to validate")
    parser.add_argument("--min-common-rows", type=int, default=1)
    parser.add_argument("--rate-tolerance", type=float, default=0.0)
    parser.add_argument("--max-latency-drop-abs-diff", type=float, default=1e-4)
    parser.add_argument("--max-api-drop-abs-diff", type=float, default=1e-4)
    parser.add_argument("--out", default="", help="Optional output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alignment_report = json.loads(Path(args.alignment_report).read_text())
    backtest_result = (
        json.loads(Path(args.backtest_result).read_text()) if args.backtest_result else None
    )
    result = evaluate_maker_acceptance(
        alignment_report,
        backtest_result=backtest_result,
        alignment_key=str(args.alignment_key),
        min_common_rows=int(args.min_common_rows),
        rate_tolerance=float(args.rate_tolerance),
        max_latency_drop_abs_diff=float(args.max_latency_drop_abs_diff),
        max_api_drop_abs_diff=float(args.max_api_drop_abs_diff),
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
