#!/usr/bin/env python3
"""Run Stage 6J-B narrow cancel-race rule replay comparisons."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analyze_cancel_fill_risk import analyze_audit_csv
from backtest_metrics import flatten_summary
from backtest_tick_mm import _load_manifest, _load_toml, run_backtest


DEFAULT_RUN_IDS = [
    "5-11-night-active",
    "5-10-day-control-1h-06",
    "5-9-noon",
    "5-9-small",
]

_BASE_RULE_CANDIDATE: dict[str, Any] = {
    "cancel_race_guard_enabled": False,
    "cancel_race_guard_pending_cancel_block": True,
    "cancel_race_guard_post_fill_cooldown_ms": 0.0,
    "adverse_timing_guard_enabled": False,
    "adverse_timing_guard_target_deterioration_enabled": True,
    "adverse_timing_guard_pending_cancel_enabled": True,
    "adverse_timing_guard_post_cancel_fill_enabled": True,
    "adverse_timing_guard_cooldown_ms": 100.0,
    "adverse_timing_guard_min_target_move_ticks": 2,
    "adverse_timing_guard_block_mode": "add_side_only",
    "add_side_toxic_timing_guard_enabled": False,
    "add_side_toxic_timing_guard_pending_cancel_enabled": True,
    "add_side_toxic_timing_guard_post_cancel_fill_enabled": True,
    "add_side_toxic_timing_guard_target_move_enabled": True,
    "add_side_toxic_timing_guard_window_ms": 100.0,
    "add_side_toxic_timing_guard_min_target_move_ticks": 2,
    "add_side_toxic_timing_guard_latency_threshold_ms": 0.0,
    "add_side_toxic_timing_guard_block_mode": "add_side_submit_only",
    "inventory_add_side_cancel_cooldown_ms": 0.0,
}


def _rule_candidate(name: str, **overrides: Any) -> dict[str, Any]:
    return {"candidate": name, **_BASE_RULE_CANDIDATE, **overrides}


RULE_CANDIDATES: list[dict[str, Any]] = [
    _rule_candidate("baseline_inflight_only"),
    _rule_candidate("add_side_guard_only", cancel_race_guard_enabled=True),
    _rule_candidate(
        "add_side_toxic_timing_50ms",
        add_side_toxic_timing_guard_enabled=True,
        add_side_toxic_timing_guard_window_ms=50.0,
    ),
    _rule_candidate(
        "add_side_toxic_timing_100ms",
        add_side_toxic_timing_guard_enabled=True,
        add_side_toxic_timing_guard_window_ms=100.0,
    ),
    _rule_candidate(
        "add_side_toxic_timing_200ms",
        add_side_toxic_timing_guard_enabled=True,
        add_side_toxic_timing_guard_window_ms=200.0,
    ),
    _rule_candidate(
        "add_side_guard_plus_toxic_timing_100ms",
        cancel_race_guard_enabled=True,
        add_side_toxic_timing_guard_enabled=True,
        add_side_toxic_timing_guard_window_ms=100.0,
    ),
    _rule_candidate(
        "broad_add_side_cooldown_200ms_control",
        inventory_add_side_cancel_cooldown_ms=200.0,
    ),
]


@dataclass(frozen=True)
class SampleInput:
    run_id: str
    run_dir: Path
    config_path: Path
    manifest_path: Path


def _write_toml_scalar(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _write_config(path: Path, cfg: dict[str, Any]) -> None:
    lines: list[str] = []
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        lines.append(f"[{section}]")
        for key, value in values.items():
            if isinstance(value, dict):
                continue
            lines.append(f"{key} = {_write_toml_scalar(value)}")
        lines.append("")
    path.write_text("\n".join(lines))


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


def _find_manifest(run_dir: Path) -> Path:
    matches = sorted((run_dir / "out" / "live_raw").glob("*/manifest_*.json"))
    if not matches:
        raise FileNotFoundError(f"live raw manifest not found under {run_dir / 'out' / 'live_raw'}")
    if len(matches) > 1:
        raise ValueError(f"multiple manifest files found for {run_dir}: {matches}")
    return matches[0]


def _load_samples(local_root: Path, run_ids: list[str]) -> list[SampleInput]:
    samples: list[SampleInput] = []
    for run_id in run_ids:
        run_dir = local_root / run_id
        config_path = run_dir / "config_backtest_audit_replay.toml"
        if not config_path.exists():
            raise FileNotFoundError(f"missing audit replay config: {config_path}")
        samples.append(
            SampleInput(
                run_id=run_id,
                run_dir=run_dir,
                config_path=config_path,
                manifest_path=_find_manifest(run_dir),
            )
        )
    return samples


def _prepare_config(base_cfg: dict[str, Any], run_out: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("paths", {})["output_root"] = str(run_out)
    cfg.setdefault("summary", {})["enabled"] = True
    cfg["summary"]["output_json"] = "summary.json"
    cfg["summary"]["daily_csv"] = "daily_summary.csv"
    cfg.setdefault("audit", {})["mode"] = "full"
    cfg["audit"]["output_csv"] = "audit_bt_stage6j.csv"

    cadence = cfg.setdefault("backtest_cadence", {})
    cadence["market_state_overlay"] = "off"
    cadence["strategy_position_overlay"] = "off"
    cadence["working_order_overlay"] = "off"

    risk = cfg.setdefault("risk", {})
    risk["inventory_inflight_exposure_enabled"] = True
    for key, value in candidate.items():
        if key == "candidate":
            continue
        risk[key] = value
    return cfg


def _safe_get(summary: dict[str, Any], key: str, default: Any = "") -> Any:
    value = summary.get(key, default)
    return default if value is None else value


def _csv_bool(row: dict[str, str], key: str) -> bool:
    raw = str(row.get(key, "")).strip().lower()
    return raw in {"1", "true", "yes"}


def _scan_action_path_coverage(audit_csv: Path) -> dict[str, int]:
    totals = {
        "add_side_submit_eligible_buy_count": 0,
        "add_side_submit_eligible_sell_count": 0,
        "add_side_submit_blocked_buy_count": 0,
        "add_side_submit_blocked_sell_count": 0,
        "add_side_submit_reduce_side_allowed_buy_count": 0,
        "add_side_submit_reduce_side_allowed_sell_count": 0,
        "add_side_submit_blocked_reduce_side_count": 0,
        "add_side_submit_blocked_total_count": 0,
    }
    with audit_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eligible_buy = _csv_bool(row, "add_side_submit_eligible_buy")
            eligible_sell = _csv_bool(row, "add_side_submit_eligible_sell")
            blocked_buy = _csv_bool(row, "add_side_submit_blocked_buy")
            blocked_sell = _csv_bool(row, "add_side_submit_blocked_sell")
            reduce_buy = _csv_bool(row, "add_side_submit_reduce_side_allowed_buy")
            reduce_sell = _csv_bool(row, "add_side_submit_reduce_side_allowed_sell")

            totals["add_side_submit_eligible_buy_count"] += int(eligible_buy)
            totals["add_side_submit_eligible_sell_count"] += int(eligible_sell)
            totals["add_side_submit_blocked_buy_count"] += int(blocked_buy)
            totals["add_side_submit_blocked_sell_count"] += int(blocked_sell)
            totals["add_side_submit_reduce_side_allowed_buy_count"] += int(reduce_buy)
            totals["add_side_submit_reduce_side_allowed_sell_count"] += int(reduce_sell)
            totals["add_side_submit_blocked_reduce_side_count"] += int(
                (blocked_buy and reduce_buy) or (blocked_sell and reduce_sell)
            )
            totals["add_side_submit_blocked_total_count"] += int(blocked_buy) + int(blocked_sell)
    return totals


def _action_contains(value: str, action: str) -> bool:
    return action in {part.strip() for part in str(value or "").split("|") if part.strip()}


def _scan_baseline_action_path_diff(baseline_csv: Path, candidate_csv: Path) -> dict[str, int]:
    totals = {
        "baseline_row_compare_count": 0,
        "baseline_row_compare_missing_count": 0,
        "baseline_row_compare_ts_mismatch_count": 0,
        "baseline_action_diff_count": 0,
        "baseline_planned_action_diff_count": 0,
        "baseline_action_or_planned_diff_count": 0,
        "baseline_submit_overlap_blocked_count": 0,
        "blocked_action_or_planned_diff_count": 0,
        "submit_removed_by_guard_count": 0,
    }
    with baseline_csv.open("r", newline="") as f_base, candidate_csv.open("r", newline="") as f_cand:
        base_reader = csv.DictReader(f_base)
        cand_reader = csv.DictReader(f_cand)
        for base_row, cand_row in itertools.zip_longest(base_reader, cand_reader):
            if base_row is None or cand_row is None:
                totals["baseline_row_compare_missing_count"] += 1
                continue
            totals["baseline_row_compare_count"] += 1
            if str(base_row.get("ts_local", "")) != str(cand_row.get("ts_local", "")):
                totals["baseline_row_compare_ts_mismatch_count"] += 1

            action_diff = str(base_row.get("action", "")) != str(cand_row.get("action", ""))
            planned_diff = str(base_row.get("planned_action", "")) != str(
                cand_row.get("planned_action", "")
            )
            any_diff = action_diff or planned_diff
            totals["baseline_action_diff_count"] += int(action_diff)
            totals["baseline_planned_action_diff_count"] += int(planned_diff)
            totals["baseline_action_or_planned_diff_count"] += int(any_diff)

            blocked_buy = _csv_bool(cand_row, "add_side_submit_blocked_buy")
            blocked_sell = _csv_bool(cand_row, "add_side_submit_blocked_sell")
            blocked = blocked_buy or blocked_sell
            totals["blocked_action_or_planned_diff_count"] += int(blocked and any_diff)

            base_action = str(base_row.get("action", ""))
            base_planned = str(base_row.get("planned_action", ""))
            cand_action = str(cand_row.get("action", ""))
            cand_planned = str(cand_row.get("planned_action", ""))
            base_buy_submit = _action_contains(base_action, "submit_buy") or _action_contains(
                base_planned, "submit_buy"
            )
            base_sell_submit = _action_contains(base_action, "submit_sell") or _action_contains(
                base_planned, "submit_sell"
            )
            cand_buy_submit = _action_contains(cand_action, "submit_buy") or _action_contains(
                cand_planned, "submit_buy"
            )
            cand_sell_submit = _action_contains(cand_action, "submit_sell") or _action_contains(
                cand_planned, "submit_sell"
            )
            buy_overlap = blocked_buy and base_buy_submit
            sell_overlap = blocked_sell and base_sell_submit
            totals["baseline_submit_overlap_blocked_count"] += int(buy_overlap) + int(sell_overlap)
            totals["submit_removed_by_guard_count"] += int(buy_overlap and not cand_buy_submit) + int(
                sell_overlap and not cand_sell_submit
            )
    return totals


def _attach_baseline_action_path_diffs(rows: list[dict[str, Any]]) -> None:
    default_counts = {
        "baseline_row_compare_count": 0,
        "baseline_row_compare_missing_count": 0,
        "baseline_row_compare_ts_mismatch_count": 0,
        "baseline_action_diff_count": 0,
        "baseline_planned_action_diff_count": 0,
        "baseline_action_or_planned_diff_count": 0,
        "baseline_submit_overlap_blocked_count": 0,
        "blocked_action_or_planned_diff_count": 0,
        "submit_removed_by_guard_count": 0,
    }
    baseline_by_run = {
        str(row.get("run_id", "")): row
        for row in rows
        if row.get("status") == "ok" and row.get("candidate") == "baseline_inflight_only"
    }
    for row in rows:
        row.update(default_counts)
        if row.get("status") != "ok":
            continue
        candidate = str(row.get("candidate", ""))
        if candidate == "baseline_inflight_only":
            row["baseline_row_compare_count"] = int(float(row.get("rows") or 0))
            continue
        base = baseline_by_run.get(str(row.get("run_id", "")))
        if base is None:
            continue
        row.update(
            _scan_baseline_action_path_diff(
                Path(str(base.get("audit_csv", ""))),
                Path(str(row.get("audit_csv", ""))),
            )
        )


def run_stage6j_b(
    *,
    local_root: Path,
    out_dir: Path,
    run_ids: list[str],
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected_candidates = candidates or RULE_CANDIDATES
    samples = _load_samples(local_root, run_ids)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "local_root": str(local_root),
        "run_ids": run_ids,
        "candidate_count": len(selected_candidates),
        "candidates": selected_candidates,
        "replay_contract": {
            "cadence": "audit_replay",
            "market_state_overlay": "off",
            "strategy_position_overlay": "off",
            "working_order_overlay": "off",
            "inventory_inflight_exposure_enabled": True,
        },
    }
    (out_dir / "stage6j_replay_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n")

    rows: list[dict[str, Any]] = []
    for sample in samples:
        base_cfg = _load_toml(sample.config_path)
        manifest = _load_manifest(sample.manifest_path)
        for idx, candidate in enumerate(selected_candidates, start=1):
            candidate_name = str(candidate["candidate"])
            run_out = out_dir / sample.run_id / f"{idx:02d}_{candidate_name}"
            run_out.mkdir(parents=True, exist_ok=True)
            cfg = _prepare_config(base_cfg, run_out, candidate)
            config_path = run_out / "config.toml"
            _write_config(config_path, cfg)

            try:
                result = run_backtest(cfg, manifest)
                summary = result["summary"]
                audit_csv = Path(str(result.get("audit_csv") or run_out / "audit_bt_stage6j.csv"))
                risk_result = analyze_audit_csv(audit_csv, run_id=f"{sample.run_id}:{candidate_name}")
                risk_summary = risk_result["summary"]

                flat_summary = flatten_summary("", summary)
                action_path_coverage = _scan_action_path_coverage(audit_csv)
                row = {
                    "run_id": sample.run_id,
                    "candidate": candidate_name,
                    "status": "ok",
                    "run_dir": str(run_out),
                    "config_path": str(config_path),
                    "summary_path": str(run_out / "summary.json"),
                    "audit_csv": str(audit_csv),
                    "cancel_race_guard_enabled": candidate["cancel_race_guard_enabled"],
                    "cancel_race_guard_pending_cancel_block": candidate[
                        "cancel_race_guard_pending_cancel_block"
                    ],
                    "cancel_race_guard_post_fill_cooldown_ms": candidate[
                        "cancel_race_guard_post_fill_cooldown_ms"
                    ],
                    "adverse_timing_guard_enabled": candidate["adverse_timing_guard_enabled"],
                    "adverse_timing_guard_cooldown_ms": candidate[
                        "adverse_timing_guard_cooldown_ms"
                    ],
                    "adverse_timing_guard_min_target_move_ticks": candidate[
                        "adverse_timing_guard_min_target_move_ticks"
                    ],
                    "add_side_toxic_timing_guard_enabled": candidate[
                        "add_side_toxic_timing_guard_enabled"
                    ],
                    "add_side_toxic_timing_guard_window_ms": candidate[
                        "add_side_toxic_timing_guard_window_ms"
                    ],
                    "add_side_toxic_timing_guard_min_target_move_ticks": candidate[
                        "add_side_toxic_timing_guard_min_target_move_ticks"
                    ],
                    "add_side_toxic_timing_guard_block_mode": candidate[
                        "add_side_toxic_timing_guard_block_mode"
                    ],
                    "inventory_add_side_cancel_cooldown_ms": candidate[
                        "inventory_add_side_cancel_cooldown_ms"
                    ],
                    "audit_replay_lag_gate_passed": result.get("audit_replay_lag_gate", {}).get("passed", ""),
                    "audit_replay_lag_gate_breach_count": result.get("audit_replay_lag_gate", {}).get(
                        "breach_count", ""
                    ),
                    "audit_replay_lag_gate_fail_count": result.get("audit_replay_lag_gate", {}).get(
                        "fail_count", ""
                    ),
                    "audit_replay_lag_gate_drop_count": result.get("audit_replay_lag_gate", {}).get(
                        "drop_count", ""
                    ),
                    "audit_replay_market_state_overlay_mode": result.get(
                        "audit_replay_market_state_overlay_mode", ""
                    ),
                    "audit_replay_strategy_position_overlay_mode": result.get(
                        "audit_replay_strategy_position_overlay_mode", ""
                    ),
                    "audit_replay_working_order_overlay_mode": result.get(
                        "audit_replay_working_order_overlay_mode", ""
                    ),
                    "cancel_fill_count": risk_summary["fill_after_cancel_request_count"],
                    "cancel_fill_notional_rate": risk_summary["fill_after_cancel_request_notional_rate"],
                    "worsening_cancel_fill_count": risk_summary[
                        "worsening_fill_after_cancel_request_count"
                    ],
                    "same_side_readd_count": risk_summary[
                        "same_side_readd_while_cancel_requested_count"
                    ],
                    "same_side_readd_then_cancel_fill_count": risk_summary[
                        "same_side_readd_then_cancel_fill_count"
                    ],
                    "source_inventory_worsening_no_readd_count": _safe_get(
                        risk_summary, "source_path_inventory_worsening_no_readd_count", 0
                    ),
                    "source_same_side_readd_inventory_worsening_count": _safe_get(
                        risk_summary, "source_path_same_side_readd_inventory_worsening_count", 0
                    ),
                    "source_inventory_reducing_cancel_race_count": _safe_get(
                        risk_summary, "source_path_inventory_reducing_cancel_race_count", 0
                    ),
                    "guard_candidate_add_side_count": _safe_get(
                        risk_summary, "guard_candidate_add_side_count", 0
                    ),
                    "guard_candidate_adverse_selection_count": _safe_get(
                        risk_summary, "guard_candidate_adverse_selection_count", 0
                    ),
                    **action_path_coverage,
                    **flat_summary,
                }
            except Exception as exc:
                row = {
                    "run_id": sample.run_id,
                    "candidate": candidate_name,
                    "status": "error",
                    "run_dir": str(run_out),
                    "config_path": str(config_path),
                    "error": repr(exc),
                    "cancel_race_guard_enabled": candidate["cancel_race_guard_enabled"],
                    "cancel_race_guard_pending_cancel_block": candidate[
                        "cancel_race_guard_pending_cancel_block"
                    ],
                    "cancel_race_guard_post_fill_cooldown_ms": candidate[
                        "cancel_race_guard_post_fill_cooldown_ms"
                    ],
                    "adverse_timing_guard_enabled": candidate["adverse_timing_guard_enabled"],
                    "adverse_timing_guard_cooldown_ms": candidate[
                        "adverse_timing_guard_cooldown_ms"
                    ],
                    "adverse_timing_guard_min_target_move_ticks": candidate[
                        "adverse_timing_guard_min_target_move_ticks"
                    ],
                    "add_side_toxic_timing_guard_enabled": candidate[
                        "add_side_toxic_timing_guard_enabled"
                    ],
                    "add_side_toxic_timing_guard_window_ms": candidate[
                        "add_side_toxic_timing_guard_window_ms"
                    ],
                    "add_side_toxic_timing_guard_min_target_move_ticks": candidate[
                        "add_side_toxic_timing_guard_min_target_move_ticks"
                    ],
                    "add_side_toxic_timing_guard_block_mode": candidate[
                        "add_side_toxic_timing_guard_block_mode"
                    ],
                    "inventory_add_side_cancel_cooldown_ms": candidate[
                        "inventory_add_side_cancel_cooldown_ms"
                    ],
                }
            rows.append(row)

    _attach_baseline_action_path_diffs(rows)
    _write_csv(out_dir / "stage6j_replay_summary.csv", rows)
    (out_dir / "stage6j_replay_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=True) + "\n"
    )

    summary = _summarize(rows)
    (out_dir / "STAGE6J_B_REPLAY_SUMMARY.md").write_text(_markdown_report(summary, rows))
    (out_dir / "stage6j_replay_decision.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n"
    )
    return {"rows": rows, "summary": summary}


def _as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        raw = row.get(key, default)
        if raw == "":
            return default
        return float(raw)
    except (TypeError, ValueError):
        return default


def _as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    return int(round(_as_float(row, key, float(default))))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    samples = sorted({str(row["run_id"]) for row in ok_rows})
    candidates = sorted({str(row["candidate"]) for row in ok_rows})
    by_candidate: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_rows = [row for row in ok_rows if row["candidate"] == candidate]
        baseline_rows = {
            str(row["run_id"]): row
            for row in ok_rows
            if row["candidate"] == "baseline_inflight_only"
        }
        max_abs_position = max((_as_float(row, "max_abs_position_notional") for row in candidate_rows), default=0.0)
        pnl_sum = sum(_as_float(row, "pnl_mtm") for row in candidate_rows)
        cancel_fill_count = sum(_as_int(row, "cancel_fill_count") for row in candidate_rows)
        inv_worsening = sum(_as_int(row, "source_inventory_worsening_no_readd_count") for row in candidate_rows)
        same_side_worsening = sum(
            _as_int(row, "source_same_side_readd_inventory_worsening_count") for row in candidate_rows
        )
        add_side_submit_blocked = sum(
            _as_int(row, "add_side_submit_blocked_total_count") for row in candidate_rows
        )
        blocked_reduce_side = sum(
            _as_int(row, "add_side_submit_blocked_reduce_side_count") for row in candidate_rows
        )
        baseline_action_or_planned_diff = sum(
            _as_int(row, "baseline_action_or_planned_diff_count") for row in candidate_rows
        )
        baseline_submit_overlap_blocked = sum(
            _as_int(row, "baseline_submit_overlap_blocked_count") for row in candidate_rows
        )
        submit_removed_by_guard = sum(
            _as_int(row, "submit_removed_by_guard_count") for row in candidate_rows
        )
        baseline_delta = []
        for row in candidate_rows:
            base = baseline_rows.get(str(row["run_id"]))
            if base is None:
                continue
            baseline_delta.append(
                {
                    "run_id": row["run_id"],
                    "pnl_delta": _as_float(row, "pnl_mtm") - _as_float(base, "pnl_mtm"),
                    "max_abs_position_delta": _as_float(row, "max_abs_position_notional")
                    - _as_float(base, "max_abs_position_notional"),
                    "cancel_fill_count_delta": _as_int(row, "cancel_fill_count")
                    - _as_int(base, "cancel_fill_count"),
                    "inventory_worsening_no_readd_delta": _as_int(
                        row, "source_inventory_worsening_no_readd_count"
                    )
                    - _as_int(base, "source_inventory_worsening_no_readd_count"),
                    "same_side_worsening_delta": _as_int(
                        row, "source_same_side_readd_inventory_worsening_count"
                    )
                    - _as_int(base, "source_same_side_readd_inventory_worsening_count"),
                    "add_side_submit_blocked_delta": _as_int(
                        row, "add_side_submit_blocked_total_count"
                    )
                    - _as_int(base, "add_side_submit_blocked_total_count"),
                }
            )
        by_candidate[candidate] = {
            "runs": len(candidate_rows),
            "pnl_mtm_sum": pnl_sum,
            "max_abs_position_notional_max": max_abs_position,
            "cancel_fill_count_sum": cancel_fill_count,
            "inventory_worsening_no_readd_count_sum": inv_worsening,
            "same_side_worsening_count_sum": same_side_worsening,
            "add_side_submit_blocked_count_sum": add_side_submit_blocked,
            "blocked_reduce_side_count_sum": blocked_reduce_side,
            "baseline_action_or_planned_diff_count_sum": baseline_action_or_planned_diff,
            "baseline_submit_overlap_blocked_count_sum": baseline_submit_overlap_blocked,
            "submit_removed_by_guard_count_sum": submit_removed_by_guard,
            "baseline_deltas": baseline_delta,
        }

    hard_failures = [
        {
            "run_id": row.get("run_id"),
            "candidate": row.get("candidate"),
            "reason": "status_error" if row.get("status") != "ok" else "gate_or_overlay",
            "error": row.get("error", ""),
        }
        for row in rows
        if row.get("status") != "ok"
        or str(row.get("audit_replay_market_state_overlay_mode", "")) != "off"
        or str(row.get("audit_replay_strategy_position_overlay_mode", "")) != "off"
        or str(row.get("audit_replay_working_order_overlay_mode", "")) != "off"
        or str(row.get("audit_replay_lag_gate_passed", "")).lower() not in {"true", "1"}
    ]
    return {
        "decision": "diagnostic_only_no_promotion",
        "sample_count": len(samples),
        "samples": samples,
        "candidate_count": len(candidates),
        "candidates": by_candidate,
        "hard_failure_count": len(hard_failures),
        "hard_failures": hard_failures,
    }


def _markdown_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage 6J-B Narrow Rule Replay Summary",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Samples: `{summary['sample_count']}`",
        f"- Candidates: `{summary['candidate_count']}`",
        f"- Hard failures: `{summary['hard_failure_count']}`",
        "",
        "## Candidate Totals",
        "",
        "| candidate | runs | pnl sum | max abs notional max | cancel-fill count | inventory worsening no readd | same-side worsening | add-side submit blocked | blocked reduce-side | baseline action diff | blocked submit overlap | submit removed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate, data in summary["candidates"].items():
        lines.append(
            "| {candidate} | {runs} | {pnl:.6f} | {max_pos:.6f} | {cancel_fill} | {inv_worse} | {same_worse} | {submit_blocked} | {blocked_reduce} | {baseline_diff} | {submit_overlap} | {submit_removed} |".format(
                candidate=candidate,
                runs=data["runs"],
                pnl=float(data["pnl_mtm_sum"]),
                max_pos=float(data["max_abs_position_notional_max"]),
                cancel_fill=int(data["cancel_fill_count_sum"]),
                inv_worse=int(data["inventory_worsening_no_readd_count_sum"]),
                same_worse=int(data["same_side_worsening_count_sum"]),
                submit_blocked=int(data["add_side_submit_blocked_count_sum"]),
                blocked_reduce=int(data["blocked_reduce_side_count_sum"]),
                baseline_diff=int(data["baseline_action_or_planned_diff_count_sum"]),
                submit_overlap=int(data["baseline_submit_overlap_blocked_count_sum"]),
                submit_removed=int(data["submit_removed_by_guard_count_sum"]),
            )
        )

    lines.extend(
        [
            "",
            "## Per-Run Results",
            "",
            "| run | candidate | pnl | max abs notional | drop api | drop latency | cancel-fill | inv-worsening no readd | same-side worsening | add-side submit blocked | blocked reduce-side | baseline action diff | blocked submit overlap | submit removed | overlays | lag gate |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        overlays = "/".join(
            str(row.get(key, ""))
            for key in [
                "audit_replay_market_state_overlay_mode",
                "audit_replay_strategy_position_overlay_mode",
                "audit_replay_working_order_overlay_mode",
            ]
        )
        lines.append(
            "| {run_id} | {candidate} | {pnl:.6f} | {max_pos:.6f} | {drop_api:.6f} | {drop_latency:.6f} | {cancel_fill} | {inv_worse} | {same_worse} | {submit_blocked} | {blocked_reduce} | {baseline_diff} | {submit_overlap} | {submit_removed} | `{overlays}` | `{lag_gate}` |".format(
                run_id=row.get("run_id", ""),
                candidate=row.get("candidate", ""),
                pnl=_as_float(row, "pnl_mtm"),
                max_pos=_as_float(row, "max_abs_position_notional"),
                drop_api=_as_float(row, "drop_api_rate"),
                drop_latency=_as_float(row, "drop_latency_rate"),
                cancel_fill=_as_int(row, "cancel_fill_count"),
                inv_worse=_as_int(row, "source_inventory_worsening_no_readd_count"),
                same_worse=_as_int(row, "source_same_side_readd_inventory_worsening_count"),
                submit_blocked=_as_int(row, "add_side_submit_blocked_total_count"),
                blocked_reduce=_as_int(row, "add_side_submit_blocked_reduce_side_count"),
                baseline_diff=_as_int(row, "baseline_action_or_planned_diff_count"),
                submit_overlap=_as_int(row, "baseline_submit_overlap_blocked_count"),
                submit_removed=_as_int(row, "submit_removed_by_guard_count"),
                overlays=overlays,
                lag_gate=row.get("audit_replay_lag_gate_passed", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is an offline rule-design replay. It is not a promotion decision.",
            "- Audit cadence is used, but market, strategy-position, and working-order overlays are forced off.",
            "- Action-path coverage columns count submit-path eligibility and blocks from audit rows; they do not prove live source-path improvement.",
            "- A candidate can only advance if it improves the targeted cancel-fill source paths without breaching position, drop, or churn constraints across current-format windows.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 6J-B cancel-race rule replay comparisons")
    parser.add_argument("--local-root", default="local_live_analysis")
    parser.add_argument("--out-dir", default="local_live_analysis/stage6j_narrow_rule_replay")
    parser.add_argument("--run-id", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_ids = list(args.run_id) if args.run_id else DEFAULT_RUN_IDS
    result = run_stage6j_b(
        local_root=Path(args.local_root).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        run_ids=run_ids,
    )
    print(
        json.dumps(
            {
                "decision": result["summary"]["decision"],
                "samples": result["summary"]["sample_count"],
                "candidates": result["summary"]["candidate_count"],
                "hard_failures": result["summary"]["hard_failure_count"],
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
