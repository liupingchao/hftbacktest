#!/usr/bin/env python3
"""Plot returns and position from audit CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _load_series(audit_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts: list[int] = []
    mid: list[float] = []
    pos: list[float] = []

    with audit_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts.append(int(float(row["ts_local"])))
                mid.append(float(row["mid"]))
                pos.append(float(row["position"]))
            except (KeyError, TypeError, ValueError):
                continue

    if not ts:
        raise ValueError(f"No valid rows in {audit_csv}")

    t = np.asarray(ts, dtype=np.int64)
    m = np.asarray(mid, dtype=np.float64)
    p = np.asarray(pos, dtype=np.float64)
    return t, m, p


def _compute_returns(mid: np.ndarray, position: np.ndarray, initial_capital: float) -> np.ndarray:
    if len(mid) <= 1:
        return np.zeros_like(mid, dtype=np.float64)

    d_mid = np.diff(mid, prepend=mid[0])
    pos_prev = np.roll(position, 1)
    pos_prev[0] = position[0]

    # Mark-to-market PnL approximation from carried inventory.
    pnl = np.cumsum(pos_prev * d_mid)
    return pnl / max(initial_capital, 1.0)


def plot(audit_csv: Path, out_dir: Path, prefix: str, initial_capital: float) -> tuple[Path, Path]:
    t, mid, pos = _load_series(audit_csv)
    ret = _compute_returns(mid, pos, initial_capital)

    x_sec = (t - t[0]) / 1_000_000_000.0

    out_dir.mkdir(parents=True, exist_ok=True)
    ret_png = out_dir / f"{prefix}_returns.png"
    pos_png = out_dir / f"{prefix}_position.png"

    plt.figure(figsize=(10, 4))
    plt.plot(x_sec, ret, linewidth=1.2)
    plt.title("Cumulative Returns (MTM Approx)")
    plt.xlabel("Seconds")
    plt.ylabel("Returns")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ret_png, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(x_sec, pos, linewidth=1.2)
    plt.title("Position")
    plt.xlabel("Seconds")
    plt.ylabel("Position")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(pos_png, dpi=150)
    plt.close()

    return ret_png, pos_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot returns and position from audit CSV")
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="audit")
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ret_png, pos_png = plot(
        audit_csv=_expand(args.audit),
        out_dir=_expand(args.out_dir),
        prefix=args.prefix,
        initial_capital=float(args.initial_capital),
    )
    print(ret_png)
    print(pos_png)


if __name__ == "__main__":
    main()
