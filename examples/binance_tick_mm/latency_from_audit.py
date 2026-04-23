#!/usr/bin/env python3
"""Generate IntpOrderLatency npz from live audit csv with spike simulation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

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

from hftbacktest.data.utils.feed_order_latency import order_latency_dtype


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _read_latency_rows(audit_csv: Path) -> np.ndarray:
    rows: list[tuple[int, int, int]] = []
    with audit_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"req_ts", "exch_ts", "resp_ts"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise KeyError(
                f"Missing required columns in {audit_csv}: {required}"
            )

        for row in reader:
            try:
                req_ts = int(row["req_ts"])
                exch_ts = int(row["exch_ts"])
                resp_ts = int(row["resp_ts"])
            except (TypeError, ValueError):
                continue
            if req_ts <= 0 or resp_ts <= req_ts:
                continue
            rows.append((req_ts, exch_ts, resp_ts))

    if not rows:
        raise ValueError(f"No valid req/exch/resp rows found in {audit_csv}")

    arr = np.array(rows, dtype=np.int64)
    arr = arr[np.argsort(arr[:, 0], kind="mergesort")]
    return arr


def build_latency_series(
    base_rows: np.ndarray,
    entry_min_ms: float,
    entry_max_ms: float,
    resp_min_ms: float,
    resp_max_ms: float,
    spike_prob: float,
    spike_min_ms: float,
    spike_max_ms: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    req_ts = base_rows[:, 0].copy()
    exch_ts = base_rows[:, 1].copy()
    resp_ts = base_rows[:, 2].copy()

    entry_ns = np.maximum(exch_ts - req_ts, 1)
    resp_ns = np.maximum(resp_ts - np.maximum(exch_ts, req_ts + 1), 1)

    entry_ns = np.clip(
        entry_ns,
        int(entry_min_ms * 1_000_000),
        int(entry_max_ms * 1_000_000),
    )
    resp_ns = np.clip(
        resp_ns,
        int(resp_min_ms * 1_000_000),
        int(resp_max_ms * 1_000_000),
    )

    rng = np.random.default_rng(seed)

    spike_mask = rng.random(len(entry_ns)) < spike_prob
    if spike_mask.any():
        entry_spike = rng.uniform(spike_min_ms, spike_max_ms, size=int(spike_mask.sum()))
        resp_spike = rng.uniform(spike_min_ms, spike_max_ms, size=int(spike_mask.sum()))
        entry_ns[spike_mask] = (entry_spike * 1_000_000).astype(np.int64)
        resp_ns[spike_mask] = (resp_spike * 1_000_000).astype(np.int64)

    out = np.zeros(len(req_ts), dtype=order_latency_dtype)
    out["req_ts"] = req_ts
    out["exch_ts"] = req_ts + entry_ns
    out["resp_ts"] = out["exch_ts"] + resp_ns
    out["_padding"] = 0

    stats = {
        "rows": float(len(out)),
        "entry_mean_ms": float(np.mean(entry_ns) / 1_000_000),
        "resp_mean_ms": float(np.mean(resp_ns) / 1_000_000),
        "entry_spike_ratio": float(np.mean(spike_mask)),
        "resp_spike_ratio": float(np.mean(spike_mask)),
        "entry_gt_5ms_ratio": float(np.mean(entry_ns > 5_000_000)),
    }

    return out, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate order latency npz from live audit csv")
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-stats", default=None)
    parser.add_argument("--entry-ms-min", type=float, default=1.2)
    parser.add_argument("--entry-ms-max", type=float, default=2.8)
    parser.add_argument("--resp-ms-min", type=float, default=1.0)
    parser.add_argument("--resp-ms-max", type=float, default=2.2)
    parser.add_argument("--spike-prob", type=float, default=0.01)
    parser.add_argument("--spike-ms-min", type=float, default=8.0)
    parser.add_argument("--spike-ms-max", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_csv = _expand(args.audit_csv)
    output_npz = _expand(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    base_rows = _read_latency_rows(audit_csv)
    out, stats = build_latency_series(
        base_rows=base_rows,
        entry_min_ms=args.entry_ms_min,
        entry_max_ms=args.entry_ms_max,
        resp_min_ms=args.resp_ms_min,
        resp_max_ms=args.resp_ms_max,
        spike_prob=args.spike_prob,
        spike_min_ms=args.spike_ms_min,
        spike_max_ms=args.spike_ms_max,
        seed=args.seed,
    )

    np.savez_compressed(output_npz, data=out)
    print(f"Saved latency npz: {output_npz}")
    print(json.dumps(stats, indent=2, ensure_ascii=True))

    if args.output_stats:
        output_stats = _expand(args.output_stats)
        output_stats.parent.mkdir(parents=True, exist_ok=True)
        output_stats.write_text(json.dumps(stats, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
