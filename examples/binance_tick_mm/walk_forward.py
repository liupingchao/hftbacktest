#!/usr/bin/env python3
"""Rolling walk-forward backtest runner for binance_tick_mm."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import tomllib
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from pipeline import _expand, prepare_range
from backtest_tick_mm import run_backtest
from plot_audit import plot as plot_audit
from run_env_test import _resolve_mac_tardis


@dataclass
class FoldRange:
    fold_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _daterange(start: date, end: date) -> list[date]:
    out: list[date] = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def _date_str(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _build_folds(start: date, end: date, train_days: int, test_days: int, max_folds: int | None) -> list[FoldRange]:
    days = _daterange(start, end)
    n = len(days)
    folds: list[FoldRange] = []

    i = 0
    fold_id = 1
    while True:
        train_lo = i
        train_hi = i + train_days - 1
        test_lo = train_hi + 1
        test_hi = test_lo + test_days - 1
        if test_hi >= n:
            break

        folds.append(
            FoldRange(
                fold_id=fold_id,
                train_start=days[train_lo],
                train_end=days[train_hi],
                test_start=days[test_lo],
                test_end=days[test_hi],
            )
        )

        if max_folds is not None and len(folds) >= max_folds:
            break

        i += test_days
        fold_id += 1

    return folds


def _safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: str, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _max_drawdown(series: list[float]) -> float:
    if not series:
        return 0.0
    peak = series[0]
    mdd = 0.0
    for x in series:
        if x > peak:
            peak = x
        dd = peak - x
        if dd > mdd:
            mdd = dd
    return mdd


def _audit_metrics(audit_csv: Path) -> dict[str, float]:
    rows: list[dict[str, str]] = []
    with audit_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {
            "rows": 0.0,
            "pnl_mtm": 0.0,
            "max_drawdown_mtm": 0.0,
            "drop_latency_rate": 0.0,
            "drop_api_rate": 0.0,
            "avg_inventory_score": 0.0,
            "avg_spread_bps": 0.0,
            "avg_vol_bps": 0.0,
            "avg_abs_position_notional": 0.0,
        }

    pnl = 0.0
    equity_curve: list[float] = [0.0]
    prev_mid = _safe_float(rows[0].get("mid", "0"))
    prev_pos = _safe_float(rows[0].get("position", "0"))

    drop_latency = 0
    drop_api = 0
    inv_sum = 0.0
    spread_sum = 0.0
    vol_sum = 0.0
    abs_pos_notional_sum = 0.0

    for row in rows:
        mid = _safe_float(row.get("mid", "0"))
        pos = _safe_float(row.get("position", "0"))

        if math.isfinite(mid) and math.isfinite(prev_mid):
            pnl += prev_pos * (mid - prev_mid)
            equity_curve.append(pnl)

        drop_latency += 1 if _safe_int(row.get("dropped_by_latency", "0")) > 0 else 0
        drop_api += 1 if _safe_int(row.get("dropped_by_api_limit", "0")) > 0 else 0
        inv_sum += _safe_float(row.get("inventory_score", "0"))
        spread_sum += _safe_float(row.get("spread_bps", "0"))
        vol_sum += _safe_float(row.get("vol_bps", "0"))
        abs_pos_notional_sum += abs(pos * mid)

        prev_mid = mid
        prev_pos = pos

    n = float(len(rows))
    return {
        "rows": n,
        "pnl_mtm": pnl,
        "max_drawdown_mtm": _max_drawdown(equity_curve),
        "drop_latency_rate": drop_latency / n,
        "drop_api_rate": drop_api / n,
        "avg_inventory_score": inv_sum / n,
        "avg_spread_bps": spread_sum / n,
        "avg_vol_bps": vol_sum / n,
        "avg_abs_position_notional": abs_pos_notional_sum / n,
    }


def _resolve_tardis_dir(target: str, cfg: dict[str, Any]) -> Path:
    if target == "mac":
        return _resolve_mac_tardis(str(cfg["paths"]["mac_tardis_dir"]))
    if target == "amdserver":
        return _expand(str(cfg["paths"]["server_tardis_dir"]))
    raise ValueError(f"Unsupported target: {target}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward runner for binance_tick_mm")
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", required=True, choices=["mac", "amdserver"])
    parser.add_argument("--start-day", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-day", required=True, help="YYYY-MM-DD")
    parser.add_argument("--train-days", type=int, default=7)
    parser.add_argument("--test-days", type=int, default=1)
    parser.add_argument("--max-folds", type=int, default=0, help="0 means no limit")
    parser.add_argument("--window", default="full_day", choices=["first_5m", "first_2h", "first_6h", "full_day"])
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--strict-timestamps", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_toml(_expand(args.config))

    train_days = int(args.train_days)
    test_days = int(args.test_days)
    if train_days <= 0 or test_days <= 0:
        raise ValueError("train_days and test_days must be positive")

    start_day = _parse_date(args.start_day)
    end_day = _parse_date(args.end_day)
    max_folds = None if int(args.max_folds) <= 0 else int(args.max_folds)

    folds = _build_folds(
        start=start_day,
        end=end_day,
        train_days=train_days,
        test_days=test_days,
        max_folds=max_folds,
    )
    if not folds:
        raise ValueError("No valid folds generated; widen date range or reduce train/test days")

    tardis_dir = _resolve_tardis_dir(args.target, cfg)
    out_root = _expand(str(cfg["paths"]["output_root"])) / "walk_forward" / args.target
    out_root.mkdir(parents=True, exist_ok=True)

    tick_size = float(cfg["market"]["tick_size"])
    lot_size = float(cfg["market"]["lot_size"])
    symbol = str(cfg["symbol"]["name"])
    snapshot_mode = str(cfg["data"].get("snapshot_mode", "ignore_sod"))
    strict_ts = bool(args.strict_timestamps) or bool(cfg["data"].get("strict_timestamps", False))

    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_dir = out_root / f"fold_{fold.fold_id:03d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_manifest = prepare_range(
            tardis_dir=tardis_dir,
            out_dir=fold_dir / "train_data",
            symbol=symbol,
            start_day=_date_str(fold.train_start),
            end_day=_date_str(fold.train_end),
            tick_size=tick_size,
            lot_size=lot_size,
            snapshot_mode=snapshot_mode,
            strict_timestamps=strict_ts,
            initial_snapshot=None,
        )

        test_manifest = prepare_range(
            tardis_dir=tardis_dir,
            out_dir=fold_dir / "test_data",
            symbol=symbol,
            start_day=_date_str(fold.test_start),
            end_day=_date_str(fold.test_end),
            tick_size=tick_size,
            lot_size=lot_size,
            snapshot_mode=snapshot_mode,
            strict_timestamps=strict_ts,
            initial_snapshot=None,
        )

        with train_manifest.open("r") as f:
            train_manifest_data = json.load(f)
        with test_manifest.open("r") as f:
            test_manifest_data = json.load(f)

        # Current implementation keeps fixed params and reports train/test metrics separately.
        train_cfg = copy.deepcopy(cfg)
        train_cfg["paths"]["output_root"] = str(fold_dir / "train_out")
        train_cfg["audit"]["output_csv"] = f"audit_train_fold_{fold.fold_id:03d}.csv"
        train_result = run_backtest(train_cfg, train_manifest_data, window_override=args.window)
        train_metrics = _audit_metrics(Path(train_result["audit_csv"]))

        test_cfg = copy.deepcopy(cfg)
        test_cfg["paths"]["output_root"] = str(fold_dir / "test_out")
        test_cfg["audit"]["output_csv"] = f"audit_test_fold_{fold.fold_id:03d}.csv"
        test_result = run_backtest(test_cfg, test_manifest_data, window_override=args.window)
        test_metrics = _audit_metrics(Path(test_result["audit_csv"]))

        ret_png = ""
        pos_png = ""
        if args.plot:
            ret, pos = plot_audit(
                audit_csv=Path(test_result["audit_csv"]),
                out_dir=fold_dir / "plots",
                prefix=f"fold_{fold.fold_id:03d}_test",
                initial_capital=float(cfg["risk"].get("max_notional_pos", 1_000_000.0)),
            )
            ret_png = str(ret)
            pos_png = str(pos)

        fold_row: dict[str, Any] = {
            "fold_id": fold.fold_id,
            "train_start": _date_str(fold.train_start),
            "train_end": _date_str(fold.train_end),
            "test_start": _date_str(fold.test_start),
            "test_end": _date_str(fold.test_end),
            "window": args.window,
            "train_audit": str(train_result["audit_csv"]),
            "test_audit": str(test_result["audit_csv"]),
            "test_returns_png": ret_png,
            "test_position_png": pos_png,
            "train_rows": int(train_metrics["rows"]),
            "test_rows": int(test_metrics["rows"]),
            "train_pnl_mtm": train_metrics["pnl_mtm"],
            "test_pnl_mtm": test_metrics["pnl_mtm"],
            "train_mdd_mtm": train_metrics["max_drawdown_mtm"],
            "test_mdd_mtm": test_metrics["max_drawdown_mtm"],
            "train_drop_latency_rate": train_metrics["drop_latency_rate"],
            "test_drop_latency_rate": test_metrics["drop_latency_rate"],
            "train_drop_api_rate": train_metrics["drop_api_rate"],
            "test_drop_api_rate": test_metrics["drop_api_rate"],
            "train_avg_inventory_score": train_metrics["avg_inventory_score"],
            "test_avg_inventory_score": test_metrics["avg_inventory_score"],
            "train_avg_spread_bps": train_metrics["avg_spread_bps"],
            "test_avg_spread_bps": test_metrics["avg_spread_bps"],
            "train_avg_vol_bps": train_metrics["avg_vol_bps"],
            "test_avg_vol_bps": test_metrics["avg_vol_bps"],
            "train_avg_abs_position_notional": train_metrics["avg_abs_position_notional"],
            "test_avg_abs_position_notional": test_metrics["avg_abs_position_notional"],
        }
        fold_rows.append(fold_row)

    summary_csv = out_root / "walk_forward_summary.csv"
    summary_json = out_root / "walk_forward_summary.json"
    _write_csv(summary_csv, fold_rows)
    summary_json.write_text(json.dumps(fold_rows, indent=2, ensure_ascii=True))

    print(
        json.dumps(
            {
                "target": args.target,
                "folds": len(fold_rows),
                "summary_csv": str(summary_csv),
                "summary_json": str(summary_json),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()

