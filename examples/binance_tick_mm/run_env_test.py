#!/usr/bin/env python3
"""Environment test harness for mac/amdserver according to the execution plan."""

from __future__ import annotations

import argparse
import copy
import json
import tomllib
from pathlib import Path

from pipeline import _expand, prepare_range
from backtest_tick_mm import run_backtest
from plot_audit import plot as plot_audit


def _load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _resolve_mac_tardis(path: str) -> Path:
    p = _expand(path)
    if p.exists():
        return p

    # Fallback ~/document -> ~/Documents typo compatibility.
    alt = Path(str(p).replace("/document/", "/Documents/")).expanduser().resolve()
    if alt.exists():
        return alt

    raise FileNotFoundError(f"Mac tardis dir not found: {p} or {alt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mac/server test for binance tick MM")
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", required=True, choices=["mac", "amdserver"])
    parser.add_argument("--day", default=None, help="Override day YYYY-MM-DD; default uses config.data.start_day")
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="For server target: use fast window fallback",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Generate returns and position figures from audit csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_toml(_expand(args.config))

    day = args.day or str(config["data"]["start_day"])
    tick_size = float(config["market"]["tick_size"])
    lot_size = float(config["market"]["lot_size"])
    symbol = str(config["symbol"]["name"])

    if args.target == "mac":
        tardis_dir = _resolve_mac_tardis(str(config["paths"]["mac_tardis_dir"]))
        window = str(config["test"].get("mac_data_window", "first_5m"))
    else:
        tardis_dir = _expand(str(config["paths"]["server_tardis_dir"]))
        if args.fast:
            window = str(config["test"].get("server_fast_window", "first_6h"))
        else:
            window = str(config["test"].get("server_data_window", "full_day"))

    output_root = _expand(str(config["paths"]["output_root"])) / args.target
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = prepare_range(
        tardis_dir=tardis_dir,
        out_dir=output_root,
        symbol=symbol,
        start_day=day,
        end_day=day,
        tick_size=tick_size,
        lot_size=lot_size,
        snapshot_mode=str(config["data"].get("snapshot_mode", "ignore_sod")),
        strict_timestamps=bool(config["data"].get("strict_timestamps", True)),
        initial_snapshot=None,
    )

    cfg = copy.deepcopy(config)
    cfg["paths"]["output_root"] = str(output_root)
    cfg["audit"]["output_csv"] = f"audit_bt_{args.target}_{day}.csv"

    manifest = json.loads(Path(manifest_path).read_text())
    result = run_backtest(cfg, manifest, window_override=window)
    result["manifest"] = str(manifest_path)
    result["window"] = window
    result["target"] = args.target

    if args.plot:
        plot_dir = output_root / "plots"
        prefix = f"{args.target}_{day}"
        ret_png, pos_png = plot_audit(
            audit_csv=Path(result["audit_csv"]),
            out_dir=plot_dir,
            prefix=prefix,
            initial_capital=float(cfg["risk"].get("max_notional_pos", 1_000_000.0)),
        )
        result["returns_png"] = str(ret_png)
        result["position_png"] = str(pos_png)

    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
