#!/usr/bin/env python3
"""Parameter-parallel exact backtest sweeps for binance_tick_mm."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from backtest_metrics import flatten_summary
from backtest_tick_mm import _expand, _load_manifest, _load_toml, run_backtest


def _set_nested(cfg: dict[str, Any], section: str, key: str, value: Any) -> None:
    if section not in cfg or not isinstance(cfg[section], dict):
        cfg[section] = {}
    cfg[section][key] = value


def _grid_items(grid: dict[str, Any]) -> list[tuple[str, str, list[Any]]]:
    items: list[tuple[str, str, list[Any]]] = []
    for section, values in grid.items():
        if not isinstance(values, dict):
            raise ValueError(f"Grid section {section!r} must be a table")
        for key, raw_values in values.items():
            if not isinstance(raw_values, list):
                raise ValueError(f"Grid value {section}.{key} must be an array")
            if not raw_values:
                raise ValueError(f"Grid value {section}.{key} must not be empty")
            items.append((section, key, raw_values))
    if not items:
        raise ValueError("Grid must define at least one parameter array")
    return items


def _param_combinations(grid: dict[str, Any]) -> list[dict[str, Any]]:
    items = _grid_items(grid)
    combos: list[dict[str, Any]] = []
    for values in itertools.product(*(item[2] for item in items)):
        combo: dict[str, Any] = {}
        for (section, key, _), value in zip(items, values, strict=True):
            combo[f"{section}.{key}"] = value
        combos.append(combo)
    return combos


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


def _run_one(args: tuple[int, dict[str, Any], dict[str, Any], dict[str, Any], str | None, str]) -> dict[str, Any]:
    run_id, base_cfg, manifest, params, window, out_dir_raw = args
    out_dir = Path(out_dir_raw) / f"run_{run_id:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(base_cfg)
    for dotted, value in params.items():
        section, key = dotted.split(".", 1)
        _set_nested(cfg, section, key, value)

    cfg.setdefault("audit", {})["mode"] = "off"
    cfg.setdefault("summary", {})["enabled"] = True
    cfg["summary"]["output_json"] = "summary.json"
    cfg["summary"]["daily_csv"] = "daily_summary.csv"
    cfg.setdefault("paths", {})["output_root"] = str(out_dir)

    config_path = out_dir / "config.toml"
    _write_config(config_path, cfg)

    try:
        result = run_backtest(cfg, manifest, window_override=window)
        summary = result["summary"]
        flat = flatten_summary("", summary)
        return {
            "run_id": run_id,
            "status": "ok",
            "run_dir": str(out_dir),
            **params,
            **flat,
        }
    except Exception as exc:
        return {
            "run_id": run_id,
            "status": "error",
            "run_dir": str(out_dir),
            "error": repr(exc),
            **params,
        }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parameter-parallel exact backtest sweeps")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--window", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fail-fast", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = _load_toml(_expand(args.base_config))
    manifest = _load_manifest(_expand(args.manifest))
    grid = _load_toml(_expand(args.grid))
    combos = _param_combinations(grid)

    if int(args.workers) <= 0:
        raise ValueError("--workers must be > 0")

    out_dir = _expand(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(i + 1, base_cfg, manifest, params, args.window, str(out_dir)) for i, params in enumerate(combos)]
    rows: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = [executor.submit(_run_one, job) for job in jobs]
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(json.dumps({"run_id": row["run_id"], "status": row["status"]}, ensure_ascii=True))
            if args.fail_fast and row["status"] != "ok":
                raise RuntimeError(f"Sweep run failed: {row}")

    rows.sort(key=lambda r: int(r["run_id"]))
    _write_rows(out_dir / "sweep_summary.csv", rows)
    (out_dir / "sweep_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=True))
    print(json.dumps({"runs": len(rows), "out": str(out_dir)}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
