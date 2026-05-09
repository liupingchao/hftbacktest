#!/usr/bin/env python3
"""Run Stage 6J-B narrow cancel-race rule replay comparisons."""

from __future__ import annotations

import argparse
import copy
import csv
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
    "5-9-noon",
    "5-9-small",
    "5-8-stage3-15m-livetest-v4",
]

RULE_CANDIDATES: list[dict[str, Any]] = [
    {
        "candidate": "baseline_inflight_only",
        "cancel_race_guard_enabled": False,
        "cancel_race_guard_pending_cancel_block": True,
        "cancel_race_guard_post_fill_cooldown_ms": 0.0,
        "inventory_add_side_cancel_cooldown_ms": 0.0,
    },
    {
        "candidate": "add_side_guard_only",
        "cancel_race_guard_enabled": True,
        "cancel_race_guard_pending_cancel_block": True,
        "cancel_race_guard_post_fill_cooldown_ms": 0.0,
        "inventory_add_side_cancel_cooldown_ms": 0.0,
    },
    {
        "candidate": "add_side_guard_post_fill_50ms",
        "cancel_race_guard_enabled": True,
        "cancel_race_guard_pending_cancel_block": True,
        "cancel_race_guard_post_fill_cooldown_ms": 50.0,
        "inventory_add_side_cancel_cooldown_ms": 0.0,
    },
    {
        "candidate": "add_side_guard_post_fill_100ms",
        "cancel_race_guard_enabled": True,
        "cancel_race_guard_pending_cancel_block": True,
        "cancel_race_guard_post_fill_cooldown_ms": 100.0,
        "inventory_add_side_cancel_cooldown_ms": 0.0,
    },
    {
        "candidate": "add_side_guard_post_fill_200ms",
        "cancel_race_guard_enabled": True,
        "cancel_race_guard_pending_cancel_block": True,
        "cancel_race_guard_post_fill_cooldown_ms": 200.0,
        "inventory_add_side_cancel_cooldown_ms": 0.0,
    },
    {
        "candidate": "broad_add_side_cooldown_200ms_control",
        "cancel_race_guard_enabled": False,
        "cancel_race_guard_pending_cancel_block": True,
        "cancel_race_guard_post_fill_cooldown_ms": 0.0,
        "inventory_add_side_cancel_cooldown_ms": 200.0,
    },
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
                    "inventory_add_side_cancel_cooldown_ms": candidate[
                        "inventory_add_side_cancel_cooldown_ms"
                    ],
                }
            rows.append(row)

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
                }
            )
        by_candidate[candidate] = {
            "runs": len(candidate_rows),
            "pnl_mtm_sum": pnl_sum,
            "max_abs_position_notional_max": max_abs_position,
            "cancel_fill_count_sum": cancel_fill_count,
            "inventory_worsening_no_readd_count_sum": inv_worsening,
            "same_side_worsening_count_sum": same_side_worsening,
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
        "| candidate | runs | pnl sum | max abs notional max | cancel-fill count | inventory worsening no readd | same-side worsening |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate, data in summary["candidates"].items():
        lines.append(
            "| {candidate} | {runs} | {pnl:.6f} | {max_pos:.6f} | {cancel_fill} | {inv_worse} | {same_worse} |".format(
                candidate=candidate,
                runs=data["runs"],
                pnl=float(data["pnl_mtm_sum"]),
                max_pos=float(data["max_abs_position_notional_max"]),
                cancel_fill=int(data["cancel_fill_count_sum"]),
                inv_worse=int(data["inventory_worsening_no_readd_count_sum"]),
                same_worse=int(data["same_side_worsening_count_sum"]),
            )
        )

    lines.extend(
        [
            "",
            "## Per-Run Results",
            "",
            "| run | candidate | pnl | max abs notional | drop api | drop latency | cancel-fill | inv-worsening no readd | same-side worsening | overlays | lag gate |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
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
            "| {run_id} | {candidate} | {pnl:.6f} | {max_pos:.6f} | {drop_api:.6f} | {drop_latency:.6f} | {cancel_fill} | {inv_worse} | {same_worse} | `{overlays}` | `{lag_gate}` |".format(
                run_id=row.get("run_id", ""),
                candidate=row.get("candidate", ""),
                pnl=_as_float(row, "pnl_mtm"),
                max_pos=_as_float(row, "max_abs_position_notional"),
                drop_api=_as_float(row, "drop_api_rate"),
                drop_latency=_as_float(row, "drop_latency_rate"),
                cancel_fill=_as_int(row, "cancel_fill_count"),
                inv_worse=_as_int(row, "source_inventory_worsening_no_readd_count"),
                same_worse=_as_int(row, "source_same_side_readd_inventory_worsening_count"),
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
