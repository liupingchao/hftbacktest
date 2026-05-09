#!/usr/bin/env python3
"""Rank and reject maker parameter sweep results."""

from __future__ import annotations

import argparse
import csv
import json
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LEGACY_MAX_ABS_POSITION_NOTIONAL = 250.0

REQUIRED_METRICS = [
    "pnl_mtm",
    "max_drawdown_mtm",
    "avg_abs_position_notional",
    "max_abs_position_notional",
    "drop_api_rate",
    "drop_latency_rate",
]


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        if raw == "":
            return default
        return float(raw)
    except (TypeError, ValueError):
        return default


def _optional_float(raw: Any) -> float | None:
    try:
        if raw is None or raw == "":
            return None
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0.0:
        return None
    return value


@dataclass(frozen=True)
class PositionLimit:
    max_abs_position_notional: float
    source: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_control_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if isinstance(data.get("summary"), dict):
        return data["summary"]
    return data


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _position_limit_from_config(
    path: Path,
    *,
    position_reference_price: float | None = None,
) -> PositionLimit | None:
    cfg = _load_toml(path)
    risk = cfg.get("risk", {})
    if not isinstance(risk, dict):
        return None

    max_notional_pos = _optional_float(risk.get("max_notional_pos"))
    max_position_qty = _optional_float(risk.get("max_position_qty"))
    limits: list[tuple[float, str]] = []
    if max_notional_pos is not None:
        limits.append((max_notional_pos, "risk.max_notional_pos"))
    if max_position_qty is not None and position_reference_price is not None:
        limits.append(
            (
                max_position_qty * position_reference_price,
                "risk.max_position_qty*position_reference_price",
            )
        )

    if not limits:
        return None

    value, source_key = min(limits, key=lambda item: item[0])
    source = f"{path}:{source_key}"
    if max_position_qty is not None and position_reference_price is None:
        source += f" (risk.max_position_qty={max_position_qty} present; no reference price)"
    return PositionLimit(max_abs_position_notional=float(value), source=source)


def resolve_position_limit(
    rows: list[dict[str, str]],
    *,
    explicit_max_abs_position_notional: float | None = None,
    base_config: Path | None = None,
    position_reference_price: float | None = None,
) -> PositionLimit:
    explicit = _optional_float(explicit_max_abs_position_notional)
    if explicit is not None:
        return PositionLimit(explicit, "cli:--max-abs-position-notional")

    if base_config is not None:
        limit = _position_limit_from_config(
            base_config,
            position_reference_price=position_reference_price,
        )
        if limit is not None:
            return limit

    for row in rows:
        config_path = str(row.get("config_path", "")).strip()
        if not config_path:
            continue
        limit = _position_limit_from_config(
            Path(config_path),
            position_reference_price=position_reference_price,
        )
        if limit is not None:
            return limit

    return PositionLimit(
        LEGACY_MAX_ABS_POSITION_NOTIONAL,
        "legacy_default:250.0",
    )


def _action_count(row: dict[str, str], action: str) -> float:
    return _safe_float(row.get(f"actions_{action}", "0"))


def _churn_count(row: dict[str, str]) -> float:
    total = 0.0
    for key, value in row.items():
        if not key.startswith("actions_"):
            continue
        action = key[len("actions_"):]
        if "submit" in action or "cancel" in action:
            total += _safe_float(value)
    return total


def _control_churn(control: dict[str, Any]) -> float:
    actions = control.get("actions", {})
    if not isinstance(actions, dict):
        return 0.0
    return sum(
        float(count)
        for action, count in actions.items()
        if "submit" in str(action) or "cancel" in str(action)
    )


def _missing_required(row: dict[str, str]) -> list[str]:
    return [key for key in REQUIRED_METRICS if str(row.get(key, "")).strip() == ""]


def _score(row: dict[str, Any], control: dict[str, Any], *, churn_weight: float) -> float:
    pnl = _safe_float(row.get("pnl_mtm"))
    drawdown = _safe_float(row.get("max_drawdown_mtm"))
    avg_abs_position = _safe_float(row.get("avg_abs_position_notional"))
    drop_api = _safe_float(row.get("drop_api_rate"))
    drop_latency = _safe_float(row.get("drop_latency_rate"))
    control_drop_api = _safe_float(control.get("drop_api_rate"))
    control_drop_latency = _safe_float(control.get("drop_latency_rate"))
    churn = _safe_float(row.get("churn_count"))
    control_churn = _safe_float(control.get("churn_count"))
    churn_penalty = churn_weight * max(0.0, churn - control_churn)
    return (
        pnl
        - drawdown
        - 0.002 * avg_abs_position
        - 50.0 * max(0.0, drop_api - control_drop_api)
        - 25.0 * max(0.0, drop_latency - control_drop_latency)
        - churn_penalty
    )


def rank_sweep(
    rows: list[dict[str, str]],
    control_summary: dict[str, Any],
    *,
    max_abs_position_notional: float,
    max_drop_api_delta: float,
    max_drop_latency_delta: float,
    min_passed_candidates: int,
    churn_weight: float,
    max_abs_position_notional_source: str = "caller",
) -> dict[str, Any]:
    control = dict(control_summary)
    control["churn_count"] = _control_churn(control)
    control_drop_api = _safe_float(control.get("drop_api_rate"))
    control_drop_latency = _safe_float(control.get("drop_latency_rate"))

    ranked: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()

    for raw in rows:
        row: dict[str, Any] = dict(raw)
        reasons: list[str] = []
        if row.get("status") != "ok":
            reasons.append("status_not_ok")

        missing = _missing_required(raw)
        if missing:
            reasons.append("missing_metrics:" + "|".join(missing))

        if str(row.get("audit_replay_lag_gate_passed", "")).lower() not in {"true", "1"}:
            reasons.append("replay_lag_gate_not_passed")

        if str(row.get("audit_replay_market_state_overlay_mode", "")) == "audit":
            reasons.append("market_state_overlay_enabled")
        if str(row.get("audit_replay_strategy_position_overlay_mode", "")) == "audit":
            reasons.append("strategy_position_overlay_enabled")
        if str(row.get("audit_replay_working_order_overlay_mode", "")) == "audit":
            reasons.append("working_order_overlay_enabled")

        max_abs_pos = _safe_float(row.get("max_abs_position_notional"))
        avg_abs_pos = _safe_float(row.get("avg_abs_position_notional"))
        drop_api = _safe_float(row.get("drop_api_rate"))
        drop_latency = _safe_float(row.get("drop_latency_rate"))
        pnl = _safe_float(row.get("pnl_mtm"))
        drawdown = _safe_float(row.get("max_drawdown_mtm"))

        row["churn_count"] = _churn_count(raw)
        row["control_churn_count"] = control["churn_count"]
        row["score"] = _score(row, control, churn_weight=churn_weight)
        row["drop_api_delta_vs_control"] = drop_api - control_drop_api
        row["drop_latency_delta_vs_control"] = drop_latency - control_drop_latency
        row["avg_abs_position_delta_vs_control"] = avg_abs_pos - _safe_float(
            control.get("avg_abs_position_notional")
        )

        if max_abs_pos > max_abs_position_notional:
            reasons.append("max_abs_position_notional")
        if drop_api > control_drop_api + max_drop_api_delta:
            reasons.append("drop_api_rate")
        if drop_latency > control_drop_latency + max_drop_latency_delta:
            reasons.append("drop_latency_rate")
        if drawdown > max(abs(pnl) * 2.0, 5.0):
            reasons.append("drawdown_vs_pnl")

        if reasons:
            row["rejection_reasons"] = "|".join(reasons)
            rejected.append(row)
            for reason in reasons:
                rejection_reasons[reason.split(":", 1)[0]] += 1
        else:
            row["rank"] = 0
            ranked.append(row)

    ranked.sort(key=lambda item: _safe_float(item.get("score")), reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx

    return {
        "passed": len(ranked) >= int(min_passed_candidates),
        "ranked": ranked,
        "rejected": rejected,
        "summary": {
            "total_runs": len(rows),
            "passed_runs": len(ranked),
            "rejected_runs": len(rejected),
            "min_passed_candidates": int(min_passed_candidates),
            "max_abs_position_notional_limit": float(max_abs_position_notional),
            "max_abs_position_notional_limit_source": max_abs_position_notional_source,
            "rejection_reasons": dict(rejection_reasons),
            "control": {
                key: control.get(key)
                for key in [
                    "rows",
                    "pnl_mtm",
                    "max_drawdown_mtm",
                    "avg_abs_position_notional",
                    "max_abs_position_notional",
                    "drop_api_rate",
                    "drop_latency_rate",
                    "actions",
                    "reject_reasons",
                    "churn_count",
                ]
            },
            "top_candidates": ranked[:5],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank maker sweep results")
    parser.add_argument("--sweep-summary", required=True)
    parser.add_argument("--control-summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-config", default="")
    parser.add_argument("--max-abs-position-notional", type=float, default=None)
    parser.add_argument("--position-reference-price", type=float, default=None)
    parser.add_argument("--max-drop-api-delta", type=float, default=0.02)
    parser.add_argument("--max-drop-latency-delta", type=float, default=0.02)
    parser.add_argument("--min-passed-candidates", type=int, default=3)
    parser.add_argument("--churn-weight", type=float, default=0.0001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_csv(Path(args.sweep_summary))
    control = _load_control_summary(Path(args.control_summary))
    position_limit = resolve_position_limit(
        rows,
        explicit_max_abs_position_notional=args.max_abs_position_notional,
        base_config=Path(args.base_config) if args.base_config else None,
        position_reference_price=args.position_reference_price,
    )
    result = rank_sweep(
        rows,
        control,
        max_abs_position_notional=position_limit.max_abs_position_notional,
        max_drop_api_delta=float(args.max_drop_api_delta),
        max_drop_latency_delta=float(args.max_drop_latency_delta),
        min_passed_candidates=int(args.min_passed_candidates),
        churn_weight=float(args.churn_weight),
        max_abs_position_notional_source=position_limit.source,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "ranked.csv", result["ranked"])
    _write_csv(out_dir / "rejected.csv", result["rejected"])
    (out_dir / "stage5_dry_run_summary.json").write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=True) + "\n"
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "passed_runs": len(result["ranked"]),
                "rejected_runs": len(result["rejected"]),
                "max_abs_position_notional_limit": position_limit.max_abs_position_notional,
                "max_abs_position_notional_limit_source": position_limit.source,
                "out": str(out_dir),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
