#!/usr/bin/env python3
"""Validate audit schema and key formula invariants."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from audit_schema import AUDIT_FIELDS


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _to_int(v: Any) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def validate(path: Path, tol: float = 1e-6) -> dict[str, Any]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = [k for k in AUDIT_FIELDS if k not in fields]
        if missing:
            raise KeyError(f"Missing columns: {missing}")

        rows = list(reader)

    bad_spread = 0
    bad_vol = 0
    bad_inventory = 0
    bad_latency_flag = 0
    bad_api_flag = 0

    latency_reasons = {"latency_guard"}
    api_reasons = {"api_interval_guard", "token_bucket", "api_limit"}

    for r in rows:
        best_bid = _to_float(r["best_bid"])
        best_ask = _to_float(r["best_ask"])
        mid = _to_float(r["mid"])
        sigma_bps = _to_float(r["vol_bps"])
        spread_bps = _to_float(r["spread_bps"])
        inv_score = _to_float(r["inventory_score"])

        if mid > 0:
            expected_spread = (best_ask - best_bid) / mid * 1e4
            if abs(expected_spread - spread_bps) > max(1e-4, tol):
                bad_spread += 1

        if not math.isfinite(sigma_bps) or sigma_bps < 0:
            bad_vol += 1

        if not (0.0 <= inv_score <= 1.0):
            bad_inventory += 1

        reason = str(r.get("reject_reason", ""))
        dropped_latency = _to_int(r.get("dropped_by_latency", 0))
        dropped_api = _to_int(r.get("dropped_by_api_limit", 0))

        if reason in latency_reasons and dropped_latency != 1:
            bad_latency_flag += 1
        if reason in api_reasons and dropped_api != 1:
            bad_api_flag += 1

    return {
        "file": str(path),
        "rows": len(rows),
        "bad_spread_formula": bad_spread,
        "bad_vol_formula": bad_vol,
        "bad_inventory_score": bad_inventory,
        "bad_latency_flag": bad_latency_flag,
        "bad_api_flag": bad_api_flag,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate audit csv schema and formulas")
    parser.add_argument("--file", required=True)
    parser.add_argument("--strict", action="store_true", default=False, help="Fail if any violation exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate(_expand(args.file))
    print(json.dumps(report, indent=2, ensure_ascii=True))

    violations = (
        report["bad_spread_formula"]
        + report["bad_vol_formula"]
        + report["bad_inventory_score"]
        + report["bad_latency_flag"]
        + report["bad_api_flag"]
    )
    if args.strict and violations > 0:
        raise SystemExit(f"Validation failed with {violations} violations")


if __name__ == "__main__":
    main()
