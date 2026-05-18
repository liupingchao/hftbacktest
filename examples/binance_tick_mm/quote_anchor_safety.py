#!/usr/bin/env python3
"""Default-off Step 5C quote-anchor safety helpers and diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_ID = "0518T004"
DEFAULT_MAX_ANCHOR_AGE_MS = 50.0
DEFAULT_TICK_SIZE = 0.1


@dataclass(frozen=True)
class QuoteAnchorSafetyConfig:
    enabled: bool = False
    max_fast_anchor_age_ms: float = DEFAULT_MAX_ANCHOR_AGE_MS
    max_depth_fallback_age_ms: float = DEFAULT_MAX_ANCHOR_AGE_MS
    allow_depth_fallback: bool = True
    suppress_on_missing_anchor: bool = True
    suppress_on_stale_anchor: bool = True

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "QuoteAnchorSafetyConfig":
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            max_fast_anchor_age_ms=max(0.0, float(cfg.get("max_fast_anchor_age_ms", DEFAULT_MAX_ANCHOR_AGE_MS))),
            max_depth_fallback_age_ms=max(0.0, float(cfg.get("max_depth_fallback_age_ms", DEFAULT_MAX_ANCHOR_AGE_MS))),
            allow_depth_fallback=bool(cfg.get("allow_depth_fallback", True)),
            suppress_on_missing_anchor=bool(cfg.get("suppress_on_missing_anchor", True)),
            suppress_on_stale_anchor=bool(cfg.get("suppress_on_stale_anchor", True)),
        )


@dataclass(frozen=True)
class QuoteAnchorSafetyResult:
    enabled: bool
    original_bid_tick: int | None
    original_ask_tick: int | None
    safe_bid_tick: int | None
    safe_ask_tick: int | None
    anchor_source: str
    anchor_bid_tick: int | None
    anchor_ask_tick: int | None
    anchor_age_ms: float
    fast_anchor_available: bool
    depth_fallback_available: bool
    depth_fallback_used: bool
    bid_clamped: bool
    ask_clamped: bool
    bid_rounding_changed: bool
    ask_rounding_changed: bool
    suppress_buy: bool
    suppress_sell: bool
    missing_anchor: bool
    stale_anchor: bool
    post_only_risk_after_recheck: bool
    diagnostic_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": int(self.enabled),
            "original_bid_tick": self.original_bid_tick if self.original_bid_tick is not None else "",
            "original_ask_tick": self.original_ask_tick if self.original_ask_tick is not None else "",
            "safe_bid_tick": self.safe_bid_tick if self.safe_bid_tick is not None else "",
            "safe_ask_tick": self.safe_ask_tick if self.safe_ask_tick is not None else "",
            "anchor_source": self.anchor_source,
            "anchor_bid_tick": self.anchor_bid_tick if self.anchor_bid_tick is not None else "",
            "anchor_ask_tick": self.anchor_ask_tick if self.anchor_ask_tick is not None else "",
            "anchor_age_ms": self.anchor_age_ms if math.isfinite(self.anchor_age_ms) else "",
            "fast_anchor_available": int(self.fast_anchor_available),
            "depth_fallback_available": int(self.depth_fallback_available),
            "depth_fallback_used": int(self.depth_fallback_used),
            "bid_clamped": int(self.bid_clamped),
            "ask_clamped": int(self.ask_clamped),
            "bid_rounding_changed": int(self.bid_rounding_changed),
            "ask_rounding_changed": int(self.ask_rounding_changed),
            "suppress_buy": int(self.suppress_buy),
            "suppress_sell": int(self.suppress_sell),
            "missing_anchor": int(self.missing_anchor),
            "stale_anchor": int(self.stale_anchor),
            "post_only_risk_after_recheck": int(self.post_only_risk_after_recheck),
            "diagnostic_reason": self.diagnostic_reason,
        }


def _valid_tick(value: int | None) -> bool:
    return value is not None and int(value) > 0


def _valid_bbo(bid_tick: int | None, ask_tick: int | None) -> bool:
    return _valid_tick(bid_tick) and _valid_tick(ask_tick) and int(ask_tick) > int(bid_tick)


def _finite_age(age_ms: float | int | None) -> float:
    if age_ms is None:
        return math.inf
    try:
        value = float(age_ms)
    except (TypeError, ValueError):
        return math.inf
    if not math.isfinite(value):
        return math.inf
    return max(0.0, value)


def _floor_tick(price: float | None, tick_size: float) -> int | None:
    if price is None or tick_size <= 0.0:
        return None
    value = float(price)
    if not math.isfinite(value) or value <= 0.0:
        return None
    return int(math.floor(value / tick_size))


def _ceil_tick(price: float | None, tick_size: float) -> int | None:
    if price is None or tick_size <= 0.0:
        return None
    value = float(price)
    if not math.isfinite(value) or value <= 0.0:
        return None
    return int(math.ceil(value / tick_size))


def _post_only_risk(
    *,
    bid_tick: int | None,
    ask_tick: int | None,
    anchor_bid_tick: int | None,
    anchor_ask_tick: int | None,
) -> bool:
    if not _valid_bbo(anchor_bid_tick, anchor_ask_tick):
        return False
    bid_risk = bid_tick is not None and int(bid_tick) > int(anchor_bid_tick)
    ask_risk = ask_tick is not None and int(ask_tick) < int(anchor_ask_tick)
    crossed = (
        bid_tick is not None
        and ask_tick is not None
        and (int(bid_tick) >= int(anchor_ask_tick) or int(ask_tick) <= int(anchor_bid_tick))
    )
    return bool(bid_risk or ask_risk or crossed)


def apply_quote_anchor_safety(
    *,
    cfg: QuoteAnchorSafetyConfig,
    target_bid_tick: int | None,
    target_ask_tick: int | None,
    tick_size: float,
    target_bid_price: float | None = None,
    target_ask_price: float | None = None,
    fast_bid_tick: int | None = None,
    fast_ask_tick: int | None = None,
    fast_anchor_age_ms: float | None = None,
    depth_bid_tick: int | None = None,
    depth_ask_tick: int | None = None,
    depth_anchor_age_ms: float | None = None,
) -> QuoteAnchorSafetyResult:
    original_bid_tick = int(target_bid_tick) if target_bid_tick is not None else None
    original_ask_tick = int(target_ask_tick) if target_ask_tick is not None else None
    if not cfg.enabled:
        return QuoteAnchorSafetyResult(
            enabled=False,
            original_bid_tick=original_bid_tick,
            original_ask_tick=original_ask_tick,
            safe_bid_tick=original_bid_tick,
            safe_ask_tick=original_ask_tick,
            anchor_source="disabled",
            anchor_bid_tick=None,
            anchor_ask_tick=None,
            anchor_age_ms=math.inf,
            fast_anchor_available=False,
            depth_fallback_available=False,
            depth_fallback_used=False,
            bid_clamped=False,
            ask_clamped=False,
            bid_rounding_changed=False,
            ask_rounding_changed=False,
            suppress_buy=False,
            suppress_sell=False,
            missing_anchor=False,
            stale_anchor=False,
            post_only_risk_after_recheck=False,
            diagnostic_reason="disabled",
        )

    rounded_bid_tick = _floor_tick(target_bid_price, tick_size)
    rounded_ask_tick = _ceil_tick(target_ask_price, tick_size)
    candidate_bid_tick = rounded_bid_tick if rounded_bid_tick is not None else original_bid_tick
    candidate_ask_tick = rounded_ask_tick if rounded_ask_tick is not None else original_ask_tick
    bid_rounding_changed = rounded_bid_tick is not None and rounded_bid_tick != original_bid_tick
    ask_rounding_changed = rounded_ask_tick is not None and rounded_ask_tick != original_ask_tick

    fast_age = _finite_age(fast_anchor_age_ms)
    depth_age = _finite_age(depth_anchor_age_ms)
    fast_bbo_valid = _valid_bbo(fast_bid_tick, fast_ask_tick)
    depth_bbo_valid = _valid_bbo(depth_bid_tick, depth_ask_tick)
    fast_fresh = fast_bbo_valid and fast_age <= cfg.max_fast_anchor_age_ms
    depth_fresh = depth_bbo_valid and depth_age <= cfg.max_depth_fallback_age_ms

    anchor_source = "missing_anchor"
    anchor_bid_tick: int | None = None
    anchor_ask_tick: int | None = None
    anchor_age_ms = math.inf
    depth_fallback_used = False
    diagnostic_reason = ""

    if fast_fresh:
        anchor_source = "bookticker"
        anchor_bid_tick = int(fast_bid_tick) if fast_bid_tick is not None else None
        anchor_ask_tick = int(fast_ask_tick) if fast_ask_tick is not None else None
        anchor_age_ms = fast_age
        diagnostic_reason = "fast_anchor"
    elif cfg.allow_depth_fallback and depth_fresh:
        anchor_source = "depth_guarded_fallback"
        anchor_bid_tick = int(depth_bid_tick) if depth_bid_tick is not None else None
        anchor_ask_tick = int(depth_ask_tick) if depth_ask_tick is not None else None
        anchor_age_ms = depth_age
        depth_fallback_used = True
        diagnostic_reason = "depth_fallback_after_fast_missing_or_stale"
    else:
        any_bbo_present = fast_bbo_valid or depth_bbo_valid
        stale_anchor = bool(any_bbo_present)
        missing_anchor = not any_bbo_present
        suppress = bool(
            (missing_anchor and cfg.suppress_on_missing_anchor)
            or (stale_anchor and cfg.suppress_on_stale_anchor)
        )
        return QuoteAnchorSafetyResult(
            enabled=True,
            original_bid_tick=original_bid_tick,
            original_ask_tick=original_ask_tick,
            safe_bid_tick=candidate_bid_tick,
            safe_ask_tick=candidate_ask_tick,
            anchor_source="stale_anchor" if stale_anchor else "missing_anchor",
            anchor_bid_tick=None,
            anchor_ask_tick=None,
            anchor_age_ms=min(fast_age, depth_age),
            fast_anchor_available=fast_bbo_valid,
            depth_fallback_available=depth_bbo_valid,
            depth_fallback_used=False,
            bid_clamped=False,
            ask_clamped=False,
            bid_rounding_changed=bid_rounding_changed,
            ask_rounding_changed=ask_rounding_changed,
            suppress_buy=suppress,
            suppress_sell=suppress,
            missing_anchor=missing_anchor,
            stale_anchor=stale_anchor,
            post_only_risk_after_recheck=False,
            diagnostic_reason="no_fresh_anchor",
        )

    safe_bid_tick = candidate_bid_tick
    safe_ask_tick = candidate_ask_tick
    bid_clamped = False
    ask_clamped = False
    if safe_bid_tick is not None and anchor_bid_tick is not None and int(safe_bid_tick) > int(anchor_bid_tick):
        safe_bid_tick = int(anchor_bid_tick)
        bid_clamped = True
    if safe_ask_tick is not None and anchor_ask_tick is not None and int(safe_ask_tick) < int(anchor_ask_tick):
        safe_ask_tick = int(anchor_ask_tick)
        ask_clamped = True

    risk_after_recheck = _post_only_risk(
        bid_tick=safe_bid_tick,
        ask_tick=safe_ask_tick,
        anchor_bid_tick=anchor_bid_tick,
        anchor_ask_tick=anchor_ask_tick,
    )
    suppress_after_risk = bool(risk_after_recheck)

    return QuoteAnchorSafetyResult(
        enabled=True,
        original_bid_tick=original_bid_tick,
        original_ask_tick=original_ask_tick,
        safe_bid_tick=safe_bid_tick,
        safe_ask_tick=safe_ask_tick,
        anchor_source=anchor_source,
        anchor_bid_tick=anchor_bid_tick,
        anchor_ask_tick=anchor_ask_tick,
        anchor_age_ms=anchor_age_ms,
        fast_anchor_available=fast_bbo_valid,
        depth_fallback_available=depth_bbo_valid,
        depth_fallback_used=depth_fallback_used,
        bid_clamped=bid_clamped,
        ask_clamped=ask_clamped,
        bid_rounding_changed=bid_rounding_changed,
        ask_rounding_changed=ask_rounding_changed,
        suppress_buy=suppress_after_risk,
        suppress_sell=suppress_after_risk,
        missing_anchor=False,
        stale_anchor=False,
        post_only_risk_after_recheck=risk_after_recheck,
        diagnostic_reason=diagnostic_reason,
    )


def _expand(path: Path) -> Path:
    return path.expanduser().resolve()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _generated_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _counter_row(name: str, value: int | float | str) -> dict[str, Any]:
    return {"metric": name, "value": value}


def _aggregate_results(results: list[QuoteAnchorSafetyResult]) -> dict[str, int]:
    return {
        "decision_rows": len(results),
        "bookticker_anchor_rows": sum(1 for row in results if row.anchor_source == "bookticker"),
        "depth_fallback_rows": sum(1 for row in results if row.depth_fallback_used),
        "missing_anchor_rows": sum(1 for row in results if row.missing_anchor),
        "stale_anchor_rows": sum(1 for row in results if row.stale_anchor),
        "bid_clamped_rows": sum(1 for row in results if row.bid_clamped),
        "ask_clamped_rows": sum(1 for row in results if row.ask_clamped),
        "bid_rounding_changed_rows": sum(1 for row in results if row.bid_rounding_changed),
        "ask_rounding_changed_rows": sum(1 for row in results if row.ask_rounding_changed),
        "suppress_buy_rows": sum(1 for row in results if row.suppress_buy),
        "suppress_sell_rows": sum(1 for row in results if row.suppress_sell),
        "post_only_risk_after_recheck_rows": sum(1 for row in results if row.post_only_risk_after_recheck),
    }


def _write_summary(path: Path, counters: dict[str, int]) -> None:
    text = f"""# 0518T004 Step 5C Quote-Anchor Safety Diagnostic

## Boundary

- Mode: default-off / diagnostic-first candidate.
- Top5 is not used as the final hard post-only anchor.
- This does not repair audit_depth/bookTicker/top5 row-exact drift.
- This does not start live, change replay lifecycle, or prove production readiness.

## Key Counts

- decision rows: `{counters.get('decision_rows', 0)}`
- bookTicker anchor rows: `{counters.get('bookticker_anchor_rows', 0)}`
- guarded depth fallback rows: `{counters.get('depth_fallback_rows', 0)}`
- missing anchor rows: `{counters.get('missing_anchor_rows', 0)}`
- stale anchor rows: `{counters.get('stale_anchor_rows', 0)}`
- bid clamped rows: `{counters.get('bid_clamped_rows', 0)}`
- ask clamped rows: `{counters.get('ask_clamped_rows', 0)}`
- suppress buy rows: `{counters.get('suppress_buy_rows', 0)}`
- suppress sell rows: `{counters.get('suppress_sell_rows', 0)}`
- post-only risk after re-check rows: `{counters.get('post_only_risk_after_recheck_rows', 0)}`

## Interpretation

The candidate safety layer uses bookTicker when fresh, guarded depth fallback when bookTicker is unavailable or stale, and otherwise suppresses fresh add-side submits. Clamp and post-clamp re-check leave zero post-only/crossed-risk rows in this diagnostic.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_quote_anchor_safety_diagnostic(
    *,
    run_dir: Path,
    output_dir: Path,
    tick_size: float | None = None,
    max_anchor_age_ms: float = DEFAULT_MAX_ANCHOR_AGE_MS,
) -> dict[str, Any]:
    from quote_anchor_diagnostic import (  # Local module import keeps this runner self-contained.
        DEFAULT_TICK_SIZE as DIAGNOSTIC_DEFAULT_TICK_SIZE,
        _config_float,
        _find_audit_csv,
        _read_config,
        load_decision_rows,
    )

    run_dir = _expand(run_dir)
    output_dir = _expand(output_dir)
    config = _read_config(run_dir)
    tick = float(tick_size or _config_float(config, "market", "tick_size", DIAGNOSTIC_DEFAULT_TICK_SIZE))
    audit_csv = _find_audit_csv(run_dir)
    sidecar_dir = run_dir / "t009_fixed_sidecar"
    joined_decisions_csv = sidecar_dir / "joined_decisions.csv"
    top5_sidecar_csv = sidecar_dir / "top5_sidecar.csv"
    decisions = load_decision_rows(
        audit_csv=audit_csv,
        joined_decisions_csv=joined_decisions_csv,
        top5_sidecar_csv=top5_sidecar_csv,
        tick_size=tick,
    )

    cfg = QuoteAnchorSafetyConfig(
        enabled=True,
        max_fast_anchor_age_ms=max_anchor_age_ms,
        max_depth_fallback_age_ms=max_anchor_age_ms,
    )
    row_dicts: list[dict[str, Any]] = []
    results: list[QuoteAnchorSafetyResult] = []
    for decision in decisions:
        fast_bid = None if decision.join_missing or decision.join_gap_crossed else decision.bookticker_bid_tick
        fast_ask = None if decision.join_missing or decision.join_gap_crossed else decision.bookticker_ask_tick
        depth_bid = None if decision.join_missing or decision.join_gap_crossed else decision.audit_bid_tick
        depth_ask = None if decision.join_missing or decision.join_gap_crossed else decision.audit_ask_tick
        result = apply_quote_anchor_safety(
            cfg=cfg,
            target_bid_tick=decision.target_bid_tick,
            target_ask_tick=decision.target_ask_tick,
            tick_size=tick,
            fast_bid_tick=fast_bid,
            fast_ask_tick=fast_ask,
            fast_anchor_age_ms=decision.bookticker_join_age_ms,
            depth_bid_tick=depth_bid,
            depth_ask_tick=depth_ask,
            depth_anchor_age_ms=decision.book_view_stale_ms,
        )
        results.append(result)
        row = {
            "strategy_seq": decision.strategy_seq,
            "ts_local": decision.ts_local,
            "join_missing": int(decision.join_missing),
            "join_gap_crossed": int(decision.join_gap_crossed),
            "join_stale": int(decision.join_stale),
            **result.as_dict(),
        }
        row_dicts.append(row)

    counters = _aggregate_results(results)
    counter_rows = [_counter_row(key, value) for key, value in counters.items()]
    changed_rows = [
        row
        for row in row_dicts
        if row["bid_clamped"]
        or row["ask_clamped"]
        or row["suppress_buy"]
        or row["suppress_sell"]
        or row["depth_fallback_used"]
        or row["missing_anchor"]
        or row["stale_anchor"]
    ]

    row_fields = [
        "strategy_seq",
        "ts_local",
        "join_missing",
        "join_gap_crossed",
        "join_stale",
        "enabled",
        "original_bid_tick",
        "original_ask_tick",
        "safe_bid_tick",
        "safe_ask_tick",
        "anchor_source",
        "anchor_bid_tick",
        "anchor_ask_tick",
        "anchor_age_ms",
        "fast_anchor_available",
        "depth_fallback_available",
        "depth_fallback_used",
        "bid_clamped",
        "ask_clamped",
        "bid_rounding_changed",
        "ask_rounding_changed",
        "suppress_buy",
        "suppress_sell",
        "missing_anchor",
        "stale_anchor",
        "post_only_risk_after_recheck",
        "diagnostic_reason",
    ]
    _write_csv(output_dir / "quote_anchor_safety_counters.csv", counter_rows, ["metric", "value"])
    _write_csv(output_dir / "quote_anchor_safety_rows.csv", row_dicts, row_fields)
    _write_csv(output_dir / "quote_anchor_safety_changed_rows.csv", changed_rows[:1000], row_fields)
    _write_summary(output_dir / "quote_anchor_safety_summary.md", counters)

    manifest = {
        "task_id": TASK_ID,
        "mode": "default_off_diagnostic_first",
        "generated_at": _generated_at(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "tick_size": tick,
        "max_anchor_age_ms": max_anchor_age_ms,
        "counters": counters,
        "inputs": {
            "audit_csv": str(audit_csv),
            "audit_csv_sha256": _hash_file(audit_csv),
            "joined_decisions_csv": str(joined_decisions_csv),
            "joined_decisions_csv_sha256": _hash_file(joined_decisions_csv),
            "top5_sidecar_csv": str(top5_sidecar_csv),
            "top5_sidecar_csv_sha256": _hash_file(top5_sidecar_csv),
        },
        "outputs": [
            "quote_anchor_safety_summary.md",
            "quote_anchor_safety_counters.csv",
            "quote_anchor_safety_rows.csv",
            "quote_anchor_safety_changed_rows.csv",
            "run_manifest.json",
        ],
        "boundary": {
            "default_behavior_changed": False,
            "top5_hard_anchor": False,
            "source_drift_repair": False,
            "fair_reservation_changed": False,
            "replay_lifecycle_changed": False,
            "live_started": False,
            "live_promotion": False,
        },
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Local live-analysis run directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for Step 5C diagnostics")
    parser.add_argument("--tick-size", type=float, default=None, help="Override tick size")
    parser.add_argument(
        "--max-anchor-age-ms",
        type=float,
        default=DEFAULT_MAX_ANCHOR_AGE_MS,
        help="Freshness threshold for bookTicker and guarded depth fallback",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = run_quote_anchor_safety_diagnostic(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        tick_size=args.tick_size,
        max_anchor_age_ms=args.max_anchor_age_ms,
    )
    print(json.dumps({"status": "ok", "output_dir": manifest["output_dir"], "counters": manifest["counters"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
