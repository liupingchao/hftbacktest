#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY_HFTBACKTEST = PROJECT_ROOT / "py-hftbacktest"
if (
    os.environ.get("HFTBACKTEST_USE_LOCAL_PY", "0") == "1"
    and PY_HFTBACKTEST.exists()
    and str(PY_HFTBACKTEST) not in sys.path
):
    sys.path.insert(0, str(PY_HFTBACKTEST))

from hftbacktest.data.utils.binancefutures import convert as convert_binancefutures_raw


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()



def _copy_streamable_gzip(input_path: Path) -> Path:
    tmp = tempfile.NamedTemporaryFile(prefix=f"{input_path.stem}_streamable_", suffix=".gz", delete=False)
    tmp_path = Path(tmp.name)
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    pending = b""
    try:
        with input_path.open("rb") as src, gzip.open(tmp, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                try:
                    out = decompressor.decompress(chunk)
                except zlib.error:
                    break
                if not out:
                    continue
                pending += out
                lines = pending.split(b"\n")
                for line in lines[:-1]:
                    dst.write(line + b"\n")
                pending = lines[-1]
            try:
                tail = decompressor.flush()
            except zlib.error:
                tail = b""
            if tail:
                pending += tail
                lines = pending.split(b"\n")
                for line in lines[:-1]:
                    dst.write(line + b"\n")
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return tmp_path


def convert_live_raw_file(
    input_gz: str | Path,
    output_npz: str | Path,
    *,
    opt: str = "",
    base_latency: float = 0.0,
    combined_stream: bool = True,
    buffer_size: int = 100_000_000,
) -> np.ndarray:
    input_path = _expand(input_gz)
    output_path = _expand(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter_input = input_path
    tmp_path: Path | None = None
    try:
        try:
            data = convert_binancefutures_raw(
                str(converter_input),
                output_filename=None,
                opt=opt,
                base_latency=base_latency,
                combined_stream=combined_stream,
                buffer_size=buffer_size,
            )
        except EOFError:
            tmp_path = _copy_streamable_gzip(input_path)
            data = convert_binancefutures_raw(
                str(tmp_path),
                output_filename=None,
                opt=opt,
                base_latency=base_latency,
                combined_stream=combined_stream,
                buffer_size=buffer_size,
            )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    np.savez_compressed(output_path, data=data)
    return data


def build_live_raw_manifest(
    *,
    symbol: str,
    start_day: str,
    end_day: str,
    data_files: list[str | Path],
    initial_snapshot: str | Path | None,
    strict_timestamps: bool,
) -> dict[str, Any]:
    return {
        "symbol": symbol.upper(),
        "start_day": start_day,
        "end_day": end_day,
        "snapshot_mode": "live_raw",
        "strict_timestamps": strict_timestamps,
        "source": "binance_live_raw",
        "days": [],
        "data_files": [str(_expand(path)) for path in data_files],
        "initial_snapshot": str(_expand(initial_snapshot)) if initial_snapshot else None,
        "latest_eod_snapshot": None,
    }


def write_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    path = _expand(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True))
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Binance live raw collector gzip to hftbacktest npz + manifest")
    parser.add_argument("--input-gz", required=True, action="append", help="Collector gzip path. Repeat for multiple files.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start-day", required=True)
    parser.add_argument("--end-day", default=None)
    parser.add_argument("--initial-snapshot", default=None)
    parser.add_argument("--strict-timestamps", action="store_true")
    parser.add_argument("--opt", default="", help="Options passed to binancefutures.convert, e.g. 't' for bookTicker events")
    parser.add_argument("--buffer-size", type=int, default=100_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol_lower = args.symbol.lower()
    end_day = args.end_day or args.start_day
    out_dir = _expand(args.out_dir) / symbol_lower
    out_dir.mkdir(parents=True, exist_ok=True)

    data_files: list[Path] = []
    for input_gz in args.input_gz:
        input_path = _expand(input_gz)
        stem = input_path.name
        if stem.endswith(".gz"):
            stem = stem[:-3]
        output_npz = out_dir / f"{stem}.npz"
        data = convert_live_raw_file(
            input_path,
            output_npz,
            opt=args.opt,
            buffer_size=args.buffer_size,
        )
        if len(data) == 0:
            raise ValueError(f"converted zero rows from {input_path}")
        data_files.append(output_npz)

    manifest = build_live_raw_manifest(
        symbol=args.symbol,
        start_day=args.start_day,
        end_day=end_day,
        data_files=data_files,
        initial_snapshot=args.initial_snapshot,
        strict_timestamps=args.strict_timestamps,
    )
    manifest_path = out_dir / f"manifest_{args.start_day}_to_{end_day}.json"
    write_manifest(manifest, manifest_path)
    print(json.dumps({"manifest": str(manifest_path), "data_files": [str(p) for p in data_files]}, indent=2))


if __name__ == "__main__":
    main()
