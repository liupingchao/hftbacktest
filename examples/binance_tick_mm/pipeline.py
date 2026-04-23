#!/usr/bin/env python3
"""Data pipeline for Binance Tardis -> hftbacktest npz with snapshot chain and manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np

# Ensure local py-hftbacktest package is importable when running from repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY_HFTBACKTEST = PROJECT_ROOT / "py-hftbacktest"
if (
    os.environ.get("HFTBACKTEST_USE_LOCAL_PY", "0") == "1"
    and PY_HFTBACKTEST.exists()
    and str(PY_HFTBACKTEST) not in sys.path
):
    sys.path.insert(0, str(PY_HFTBACKTEST))

from hftbacktest.data.utils.tardis import convert
from hftbacktest.data.utils.snapshot import create_last_snapshot
from hftbacktest.types import EXCH_EVENT, LOCAL_EVENT


@dataclass
class DayArtifacts:
    day: str
    trades_src: str
    depth_src: str
    data_npz: str
    sod_snapshot: str | None
    eod_snapshot: str


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _iter_days(start_day: str, end_day: str) -> Iterable[date]:
    start = datetime.strptime(start_day, "%Y-%m-%d").date()
    end = datetime.strptime(end_day, "%Y-%m-%d").date()
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def _day_tokens(day: date) -> tuple[str, str]:
    return day.strftime("%Y-%m-%d"), day.strftime("%Y%m%d")


def _find_tardis_file(base_dir: Path, symbol: str, day: date, token: str, allow_undated: bool) -> Path:
    day_dash, day_compact = _day_tokens(day)
    day_slash = day.strftime("%Y/%m/%d")
    dated_candidates: list[Path] = []
    undated_candidates: list[Path] = []
    for path in base_dir.rglob("*.csv*"):
        name = path.name.lower()
        path_text = str(path).lower()
        if token.lower() not in path_text:
            continue
        if symbol.lower() not in name:
            continue
        if day_dash in path_text or day_compact in path_text or day_slash in path_text:
            dated_candidates.append(path)
        else:
            undated_candidates.append(path)

    if dated_candidates:
        # Prefer gzip csv when both exist.
        dated_candidates.sort(key=lambda p: (0 if p.name.endswith(".csv.gz") else 1, len(str(p))))
        return dated_candidates[0]

    if allow_undated and len(undated_candidates) == 1:
        return undated_candidates[0]
    if allow_undated and len(undated_candidates) > 1:
        sample = ", ".join(str(p) for p in undated_candidates[:3])
        raise FileNotFoundError(
            f"Multiple undated Tardis {token} files for {symbol} under {base_dir}; "
            f"please provide day-suffixed files or keep only one. Samples: {sample}"
        )

    raise FileNotFoundError(
        f"No day-matched Tardis {token} file found for {symbol} {day.isoformat()} under {base_dir}"
    )


def _strict_timestamp_report(data: np.ndarray) -> dict[str, int]:
    exch_mask = (data["ev"] & EXCH_EVENT) == EXCH_EVENT
    local_mask = (data["ev"] & LOCAL_EVENT) == LOCAL_EVENT

    exch_ts = data["exch_ts"][exch_mask]
    local_ts = data["local_ts"][local_mask]

    exch_non_strict = int(np.sum(np.diff(exch_ts) <= 0)) if len(exch_ts) > 1 else 0
    local_non_strict = int(np.sum(np.diff(local_ts) <= 0)) if len(local_ts) > 1 else 0

    return {
        "rows": int(len(data)),
        "exch_rows": int(len(exch_ts)),
        "local_rows": int(len(local_ts)),
        "exch_non_strict_pairs": exch_non_strict,
        "local_non_strict_pairs": local_non_strict,
    }


def _enforce_strict(data: np.ndarray, context: str, report_path: Path | None = None) -> None:
    report = _strict_timestamp_report(data)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))
    if report["exch_non_strict_pairs"] > 0 or report["local_non_strict_pairs"] > 0:
        raise ValueError(
            f"Strict timestamp validation failed ({context}). Report: {report_path or report}"
        )


def convert_one_day(
    tardis_dir: Path,
    out_dir: Path,
    symbol: str,
    day: date,
    tick_size: float,
    lot_size: float,
    snapshot_mode: str,
    strict_timestamps: bool,
    sod_snapshot: str | None,
    allow_undated: bool,
) -> DayArtifacts:
    day_tag = day.strftime("%Y%m%d")
    trades_file = _find_tardis_file(tardis_dir, symbol, day, "trades", allow_undated)
    depth_file = _find_tardis_file(tardis_dir, symbol, day, "incremental_book_l2", allow_undated)

    day_out = out_dir / symbol.lower()
    day_out.mkdir(parents=True, exist_ok=True)

    data_npz = day_out / f"{symbol.lower()}_{day_tag}.npz"
    eod_snapshot = day_out / f"snapshot_{symbol.lower()}_{day_tag}_eod.npz"
    strict_report = day_out / f"strict_report_{symbol.lower()}_{day_tag}.json"

    data = convert(
        input_files=[str(trades_file), str(depth_file)],
        output_filename=None,
        snapshot_mode=snapshot_mode,
    )

    if strict_timestamps:
        _enforce_strict(data, f"{symbol} {day.isoformat()}", strict_report)

    np.savez_compressed(data_npz, data=data)

    create_last_snapshot(
        data=[str(data_npz)],
        tick_size=tick_size,
        lot_size=lot_size,
        initial_snapshot=sod_snapshot,
        output_snapshot_filename=str(eod_snapshot),
    )

    return DayArtifacts(
        day=day.isoformat(),
        trades_src=str(trades_file),
        depth_src=str(depth_file),
        data_npz=str(data_npz),
        sod_snapshot=sod_snapshot,
        eod_snapshot=str(eod_snapshot),
    )


def prepare_range(
    tardis_dir: Path,
    out_dir: Path,
    symbol: str,
    start_day: str,
    end_day: str,
    tick_size: float,
    lot_size: float,
    snapshot_mode: str,
    strict_timestamps: bool,
    initial_snapshot: str | None,
) -> Path:
    artifacts: list[DayArtifacts] = []
    sod_snapshot = initial_snapshot
    days = list(_iter_days(start_day, end_day))
    allow_undated = len(days) == 1

    for day in days:
        item = convert_one_day(
            tardis_dir=tardis_dir,
            out_dir=out_dir,
            symbol=symbol,
            day=day,
            tick_size=tick_size,
            lot_size=lot_size,
            snapshot_mode=snapshot_mode,
            strict_timestamps=strict_timestamps,
            sod_snapshot=sod_snapshot,
            allow_undated=allow_undated,
        )
        artifacts.append(item)
        sod_snapshot = item.eod_snapshot

    manifest = {
        "symbol": symbol,
        "start_day": start_day,
        "end_day": end_day,
        "snapshot_mode": snapshot_mode,
        "strict_timestamps": strict_timestamps,
        "days": [item.__dict__ for item in artifacts],
        "data_files": [item.data_npz for item in artifacts],
        "initial_snapshot": artifacts[0].sod_snapshot if artifacts else None,
        "latest_eod_snapshot": artifacts[-1].eod_snapshot if artifacts else None,
    }

    manifest_dir = out_dir / symbol.lower()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"manifest_{start_day}_to_{end_day}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True))
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Binance Tardis data for hftbacktest")
    parser.add_argument("--tardis-dir", required=True, help="Directory containing raw Tardis CSV/CSV.GZ files")
    parser.add_argument("--out-dir", required=True, help="Directory to write npz/snapshot/manifest")
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. BTCUSDT")
    parser.add_argument("--start-day", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-day", required=True, help="YYYY-MM-DD")
    parser.add_argument("--tick-size", type=float, required=True)
    parser.add_argument("--lot-size", type=float, required=True)
    parser.add_argument(
        "--snapshot-mode",
        default="ignore_sod",
        choices=["process", "ignore_sod", "ignore"],
    )
    parser.add_argument("--initial-snapshot", default=None, help="Optional SOD snapshot npz for start day")
    parser.add_argument(
        "--strict-timestamps",
        action="store_true",
        default=False,
        help="Fail fast if exch/local timestamps are not strictly increasing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tardis_dir = _expand(args.tardis_dir)
    out_dir = _expand(args.out_dir)
    initial_snapshot = str(_expand(args.initial_snapshot)) if args.initial_snapshot else None

    manifest_path = prepare_range(
        tardis_dir=tardis_dir,
        out_dir=out_dir,
        symbol=args.symbol,
        start_day=args.start_day,
        end_day=args.end_day,
        tick_size=args.tick_size,
        lot_size=args.lot_size,
        snapshot_mode=args.snapshot_mode,
        strict_timestamps=args.strict_timestamps,
        initial_snapshot=initial_snapshot,
    )
    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
