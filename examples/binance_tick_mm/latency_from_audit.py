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
            event_type = str(row.get("event_type", "")).strip()
            if event_type not in {"", "0", "decision"}:
                continue
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


def _distribution_ms(ns: np.ndarray) -> dict[str, float]:
    if len(ns) == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    ms = ns.astype(np.float64) / 1_000_000.0
    return {
        "mean": float(np.mean(ms)),
        "p50": float(np.quantile(ms, 0.50)),
        "p90": float(np.quantile(ms, 0.90)),
        "p99": float(np.quantile(ms, 0.99)),
        "max": float(np.max(ms)),
    }


def _latency_stats(entry_ns: np.ndarray, resp_ns: np.ndarray, *, mode: str) -> dict[str, float | str]:
    entry = _distribution_ms(entry_ns)
    resp = _distribution_ms(resp_ns)
    return {
        "mode": mode,
        "rows": float(len(entry_ns)),
        "entry_mean_ms": entry["mean"],
        "entry_p50_ms": entry["p50"],
        "entry_p90_ms": entry["p90"],
        "entry_p99_ms": entry["p99"],
        "entry_max_ms": entry["max"],
        "resp_mean_ms": resp["mean"],
        "resp_p50_ms": resp["p50"],
        "resp_p90_ms": resp["p90"],
        "resp_p99_ms": resp["p99"],
        "resp_max_ms": resp["max"],
        "entry_gt_5ms_ratio": float(np.mean(entry_ns > 5_000_000)) if len(entry_ns) else 0.0,
    }


def _latency_array(req_ts: np.ndarray, entry_ns: np.ndarray, resp_ns: np.ndarray) -> np.ndarray:
    out = np.zeros(len(req_ts), dtype=order_latency_dtype)
    out["req_ts"] = req_ts
    out["exch_ts"] = req_ts + entry_ns
    out["resp_ts"] = out["exch_ts"] + resp_ns
    out["_padding"] = 0
    return out


def build_observed_latency_series(base_rows: np.ndarray) -> tuple[np.ndarray, dict[str, float | str]]:
    req_ts = base_rows[:, 0].copy()
    exch_ts = base_rows[:, 1].copy()
    resp_ts = base_rows[:, 2].copy()

    valid = (req_ts > 0) & (exch_ts > req_ts) & (resp_ts > exch_ts)
    req_ts = req_ts[valid]
    exch_ts = exch_ts[valid]
    resp_ts = resp_ts[valid]
    if len(req_ts) == 0:
        raise ValueError("No valid observed latency rows after filtering")

    entry_ns = exch_ts - req_ts
    resp_ns = resp_ts - exch_ts
    out = np.zeros(len(req_ts), dtype=order_latency_dtype)
    out["req_ts"] = req_ts
    out["exch_ts"] = exch_ts
    out["resp_ts"] = resp_ts
    out["_padding"] = 0
    return out, _latency_stats(entry_ns, resp_ns, mode="observed")


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
) -> tuple[np.ndarray, dict[str, float | str]]:
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

    out = _latency_array(req_ts, entry_ns, resp_ns)
    stats = _latency_stats(entry_ns, resp_ns, mode="synthetic")
    stats["entry_spike_ratio"] = float(np.mean(spike_mask))
    stats["resp_spike_ratio"] = float(np.mean(spike_mask))

    return out, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate order latency npz from live audit csv")
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-stats", default=None)
    parser.add_argument("--mode", choices=["synthetic", "observed"], default="synthetic")
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
    if args.mode == "observed":
        out, stats = build_observed_latency_series(base_rows)
    else:
        out, stats = build_latency_series(
            base_rows=base_rows,
            entry_min_ms=args.entry_ms_min,
            entry_max_ms=args.entry_ms_max,
            resp_min_ms=args.resp_ms_min,
            resp_max_ms=args.resp_ms_max,
            spike_prob=args.spike_prob,
            spike_min_ms=args.spike_ms_min,
            spike_max_ms=args.spike_max_ms,
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
