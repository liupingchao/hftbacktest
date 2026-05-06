#!/usr/bin/env python3
"""Orchestrate local live/backtest alignment for a Binance tick-mm run.

This script implements the post-live workflow:

1. Fetch run artifacts from admin@awsserver1:/home/admin/hft_live/runs/<RUN_ID>.
2. Generate IntpOrderLatency data from audit_live.csv.
3. Convert live collector gzip files to hftbacktest NPZ + manifest.
4. Run same-window backtests in normal cadence and audit_replay cadence.
5. Compare audit_bt against audit_live.
6. Write a summary and archive the run locally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
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

from backtest_tick_mm import run_backtest
from compare_audit import compare
from latency_from_audit import _read_latency_rows, build_latency_series, build_observed_latency_series
from pipeline_live_raw import build_live_raw_manifest, convert_live_raw_file, write_manifest


@dataclass
class LiveWindow:
    first_ts_local: int
    last_ts_local: int
    rows: int
    start_day: str
    end_day: str


@dataclass
class LiveRunIdentity:
    requested_run_id: str
    audit_run_id: str


@dataclass
class LiveInitialState:
    position: float
    rest_position: float | None
    ts_local: int
    source_field: str


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _run(cmd: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    print("+", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if value is None:
        return '""'
    return str(value)


def _write_toml(path: Path, cfg: dict[str, Any]) -> None:
    lines: list[str] = []
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        lines.append(f"[{section}]")
        for key, value in values.items():
            if isinstance(value, dict):
                continue
            lines.append(f"{key} = {_toml_scalar(value)}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _copy_first_existing(candidates: list[Path], dest: Path) -> Path | None:
    for src in candidates:
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            return dest
    return None


def fetch_remote_artifacts(
    *,
    run_id: str,
    local_dir: Path,
    remote_host: str,
    remote_root: str,
    dry_run: bool,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "logs").mkdir(parents=True, exist_ok=True)
    (local_dir / "raw_market_data").mkdir(parents=True, exist_ok=True)

    remote_run = f"{remote_host}:{remote_root}/runs/{run_id}"
    fetches = [
        (f"{remote_run}/config_live.toml", str(local_dir / "config_live.toml")),
        (f"{remote_run}/binancefutures.toml", str(local_dir / "binancefutures.toml")),
        (f"{remote_run}/logs/*.log", str(local_dir / "logs/")),
        (f"{remote_run}/data/*.gz", str(local_dir / "raw_market_data/")),
        (f"{remote_run}/output/audit_live*.csv", str(local_dir / "")),
        (
            f"{remote_host}:{remote_root}/hftbacktest/examples/binance_tick_mm/audit_live*.csv",
            str(local_dir / ""),
        ),
    ]

    for src, dest in fetches:
        try:
            _run(["scp", src, dest], dry_run=dry_run)
        except subprocess.CalledProcessError as exc:
            if "audit_live" in src:
                print(f"warning: optional fetch failed: {src}: {exc}")
                continue
            raise


def find_audit_csv(local_dir: Path, run_id: str) -> Path:
    matches = sorted(local_dir.glob("audit_live*.csv"))
    if not matches:
        raise FileNotFoundError(f"live audit CSV not found: {local_dir}/audit_live*.csv")
    run_matches = [p for p in matches if run_id in p.name]
    if run_matches:
        return sorted(run_matches)[0]
    if len(matches) == 1:
        return matches[0]
    return max(matches, key=lambda p: p.stat().st_mtime)


def read_live_window(audit_csv: Path) -> LiveWindow:
    ts: list[int] = []
    with audit_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "ts_local" not in set(reader.fieldnames or []):
            raise KeyError(f"missing ts_local in {audit_csv}")
        for row in reader:
            raw = str(row.get("ts_local", "")).strip()
            if not raw:
                continue
            try:
                ts.append(int(raw))
            except ValueError:
                ts.append(int(float(raw)))
    if not ts:
        raise ValueError(f"no valid ts_local rows in {audit_csv}")

    first = min(ts)
    last = max(ts)
    start_day = datetime.fromtimestamp(first / 1_000_000_000.0, tz=timezone.utc).strftime(
        "%Y-%m-%d"
    )
    end_day = datetime.fromtimestamp(last / 1_000_000_000.0, tz=timezone.utc).strftime(
        "%Y-%m-%d"
    )
    return LiveWindow(first, last, len(ts), start_day, end_day)


def read_live_audit_run_id(audit_csv: Path, requested_run_id: str) -> LiveRunIdentity:
    seen: list[str] = []
    with audit_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "run_id" not in set(reader.fieldnames or []):
            return LiveRunIdentity(requested_run_id, requested_run_id)
        for row in reader:
            run_id = str(row.get("run_id", "")).strip()
            if run_id and run_id not in seen:
                seen.append(run_id)

    if not seen:
        return LiveRunIdentity(requested_run_id, requested_run_id)
    if len(seen) == 1:
        return LiveRunIdentity(requested_run_id, seen[0])

    prefix_matches = [run_id for run_id in seen if run_id.startswith(requested_run_id)]
    if len(prefix_matches) == 1:
        return LiveRunIdentity(requested_run_id, prefix_matches[0])

    candidates = prefix_matches or seen
    raise ValueError(
        "multiple live audit run_id values found; cannot choose automatically: "
        + ", ".join(candidates)
    )


def _row_float(row: dict[str, str], key: str) -> float | None:
    raw = str(row.get(key, "")).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _row_int(row: dict[str, str], key: str) -> int:
    raw = str(row.get(key, "")).strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        return int(float(raw))


def read_live_initial_state(audit_csv: Path) -> LiveInitialState:
    with audit_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if "position" not in fields and "rest_position" not in fields:
            raise KeyError(f"missing position/rest_position in {audit_csv}")
        for row in reader:
            position = _row_float(row, "position")
            rest_position = _row_float(row, "rest_position")
            if rest_position is not None:
                return LiveInitialState(
                    position=rest_position,
                    rest_position=rest_position,
                    ts_local=_row_int(row, "ts_local"),
                    source_field="rest_position",
                )
            if position is not None:
                return LiveInitialState(
                    position=position,
                    rest_position=None,
                    ts_local=_row_int(row, "ts_local"),
                    source_field="position",
                )
    raise ValueError(f"no valid initial position found in {audit_csv}")


def read_live_first_feed_ts_local(audit_csv: Path, run_id: str = "") -> int:
    with audit_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if "ts_local" not in fields:
            raise KeyError(f"missing ts_local in {audit_csv}")
        has_ts_exch = "ts_exch" in fields
        has_feed_latency = "feed_latency_ns" in fields
        for row in reader:
            if run_id and row.get("run_id", "") != run_id:
                continue
            event_type = str(row.get("event_type", "")).strip().lower()
            if event_type not in {"", "0", "decision"}:
                continue
            ts_exch = _row_int(row, "ts_exch") if has_ts_exch else 0
            feed_latency_ns = _row_int(row, "feed_latency_ns") if has_feed_latency else 0
            if ts_exch > 0 and feed_latency_ns >= 0:
                return ts_exch + feed_latency_ns
            ts_local = _row_int(row, "ts_local")
            if ts_local > 0:
                return ts_local
    raise ValueError(f"no valid decision feed timestamp found in {audit_csv}")


def generate_latency(
    *,
    audit_csv: Path,
    output_npz: Path,
    output_stats: Path,
    latency_mode: str,
    entry_min_ms: float,
    entry_max_ms: float,
    resp_min_ms: float,
    resp_max_ms: float,
    spike_prob: float,
    spike_min_ms: float,
    spike_max_ms: float,
    seed: int,
) -> dict[str, Any]:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    base_rows = _read_latency_rows(audit_csv)
    if latency_mode == "observed":
        out, stats = build_observed_latency_series(base_rows)
    elif latency_mode == "synthetic":
        out, stats = build_latency_series(
            base_rows=base_rows,
            entry_min_ms=entry_min_ms,
            entry_max_ms=entry_max_ms,
            resp_min_ms=resp_min_ms,
            resp_max_ms=resp_max_ms,
            spike_prob=spike_prob,
            spike_min_ms=spike_min_ms,
            spike_max_ms=spike_max_ms,
            seed=seed,
        )
    else:
        raise ValueError(f"unsupported latency_mode: {latency_mode}")
    np.savez_compressed(output_npz, data=out)
    output_stats.write_text(json.dumps(stats, indent=2, ensure_ascii=True))
    return stats


def convert_raw_market_data(
    *,
    raw_files: list[Path],
    out_dir: Path,
    symbol: str,
    window: LiveWindow,
    buffer_size: int,
    opt: str,
) -> Path:
    symbol_dir = out_dir / symbol.lower()
    symbol_dir.mkdir(parents=True, exist_ok=True)
    data_files: list[Path] = []

    for raw in raw_files:
        stem = raw.name[:-3] if raw.name.endswith(".gz") else raw.name
        output_npz = symbol_dir / f"{stem}.npz"
        data = convert_live_raw_file(raw, output_npz, opt=opt, buffer_size=buffer_size)
        if len(data) == 0:
            raise ValueError(f"converted zero rows from {raw}")
        data_files.append(output_npz)

    manifest = build_live_raw_manifest(
        symbol=symbol,
        start_day=window.start_day,
        end_day=window.end_day,
        data_files=data_files,
        initial_snapshot=None,
        strict_timestamps=False,
    )
    manifest_path = symbol_dir / f"manifest_{window.start_day}_to_{window.end_day}.json"
    return write_manifest(manifest, manifest_path)


def build_backtest_config(
    *,
    base_cfg: dict[str, Any],
    local_dir: Path,
    latency_npz: Path,
    audit_csv: Path,
    run_id: str,
    mode: str,
    initial_state: LiveInitialState | None = None,
) -> tuple[dict[str, Any], Path]:
    cfg = deepcopy(base_cfg)
    cfg.setdefault("paths", {})["output_root"] = str(local_dir / "out" / f"backtest_{mode}")
    cfg.setdefault("latency", {})["order_latency_npz"] = str(latency_npz)
    cfg.setdefault("latency", {}).setdefault("latency_guard_ms", 5.0)
    cfg["feed_latency"] = {
        "audit_csv": str(audit_csv),
        "run_id": run_id,
    }
    cfg.setdefault("queue", {})["power_prob_n"] = float(cfg.get("queue", {}).get("power_prob_n", 5.0))
    cfg.setdefault("audit", {})["output_csv"] = f"audit_bt_{mode}.csv"
    cfg["audit"].setdefault("flush_every", 100)
    cfg.setdefault("summary", {})["enabled"] = True
    cfg["summary"]["output_json"] = f"summary_{mode}.json"
    cfg["summary"]["daily_csv"] = f"daily_summary_{mode}.csv"
    cfg.setdefault("backtest", {})["window"] = "full_day"
    cfg["backtest"].setdefault("wait_timeout_ns", 1_000_000)
    cfg["market_data_replay"] = {
        "live_local_feed_compat": True,
    }
    init_enabled = initial_state is not None and abs(float(initial_state.position)) > 0.0
    cfg["alignment_init"] = {
        "enabled": init_enabled,
        "position_mode": "synthetic_fill" if init_enabled else "off",
        "position": float(initial_state.position) if initial_state else 0.0,
        "rest_position": (
            float(initial_state.rest_position)
            if initial_state is not None and initial_state.rest_position is not None
            else ""
        ),
        "source": initial_state.source_field if initial_state else "",
        "ts_local": int(initial_state.ts_local) if initial_state else 0,
        "order_id_start": 900_000_000_000,
    }

    if mode == "normal":
        cfg["backtest_cadence"] = {
            "mode": "fixed_interval",
            "enabled": False,
            "min_decision_interval_ms": 0.0,
            "audit_csv": "",
            "run_id": "",
            "ts_column": "ts_local",
            "tolerance_ms": 0.0,
        }
    elif mode == "audit_replay":
        cfg["backtest_cadence"] = {
            "mode": "audit_replay",
            "enabled": True,
            "audit_csv": str(audit_csv),
            "run_id": run_id,
            "ts_column": "ts_local",
            "tolerance_ms": 0.0,
            "replay_mode": "emit_due",
            "trigger_ts_source": "feed_local",
            "feed_latency_column": "feed_latency_ns",
            "max_lag_ms": 250.0,
            "max_exch_lag_ms": 250.0,
            "strict_lag_gate": True,
            "lag_gate_action": "fail",
            "market_state_overlay": "audit",
        }
    else:
        raise ValueError(f"unsupported backtest mode: {mode}")

    path = local_dir / f"config_backtest_{mode}.toml"
    _write_toml(path, cfg)
    return cfg, path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def run_alignment_backtests(
    *,
    base_cfg: dict[str, Any],
    local_dir: Path,
    run_id: str,
    audit_csv: Path,
    latency_npz: Path,
    manifest_path: Path,
    window: LiveWindow,
    initial_state: LiveInitialState | None,
    skip_backtest: bool,
    audit_replay_prewarm_ms: float,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    first_feed_ts_local = read_live_first_feed_ts_local(audit_csv, run_id)
    for mode in ("normal", "audit_replay"):
        cfg, cfg_path = build_backtest_config(
            base_cfg=base_cfg,
            local_dir=local_dir,
            latency_npz=latency_npz,
            audit_csv=audit_csv,
            run_id=run_id,
            mode=mode,
            initial_state=initial_state,
        )
        if skip_backtest:
            results[mode] = {"config": str(cfg_path), "skipped": True}
            continue

        manifest = json.loads(manifest_path.read_text())
        failed_error = ""
        try:
            slice_start = window.first_ts_local
            if mode == "audit_replay":
                prewarm_ns = int(float(audit_replay_prewarm_ms) * 1_000_000)
                slice_start = max(0, first_feed_ts_local - prewarm_ns)
            result = run_backtest(
                config=cfg,
                manifest=manifest,
                window_override="full_day",
                slice_ts_local_start=slice_start,
                slice_ts_local_end=window.last_ts_local,
            )
        except RuntimeError as exc:
            failed_error = str(exc)
            summary_path = Path(str(cfg["paths"]["output_root"])) / str(
                cfg.get("summary", {}).get("output_json", f"summary_{mode}.json")
            )
            if not summary_path.exists():
                raise
            result = json.loads(summary_path.read_text())
            result["failed"] = True
            result["error"] = failed_error
            result["traceback"] = traceback.format_exc()
        result["config"] = str(cfg_path)
        result["decision_first_ts_local"] = int(window.first_ts_local)
        result["first_feed_ts_local"] = int(first_feed_ts_local)
        result["audit_replay_prewarm_ms"] = (
            float(audit_replay_prewarm_ms) if mode == "audit_replay" else 0.0
        )
        write_json(local_dir / f"backtest_{mode}_result.json", result)

        bt_audit = Path(str(cfg["paths"]["output_root"])) / str(cfg["audit"]["output_csv"])
        report = compare(bt_audit, audit_csv)
        report_path = local_dir / f"alignment_report_{mode}.json"
        write_json(report_path, report)
        result["alignment_report"] = str(report_path)
        if failed_error:
            result["failed"] = True
            result["error"] = failed_error
        results[mode] = result
    return results


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_run(local_dir: Path, archive_root: Path) -> Path:
    manifest_path = local_dir / "FILE_MANIFEST.txt"
    checksums_path = local_dir / "SHA256SUMS.txt"
    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    manifest_path.write_text("\n".join(str(p.relative_to(local_dir)) for p in files) + "\n")

    checksum_lines = []
    for p in files:
        if p == checksums_path:
            continue
        checksum_lines.append(f"{sha256_file(p)}  {p.relative_to(local_dir)}")
    checksums_path.write_text("\n".join(checksum_lines) + "\n")

    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = archive_root / f"{local_dir.name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(local_dir, arcname=local_dir.name)

    checksum_path = archive_root / f"{archive_path.name}.sha256"
    checksum_path.write_text(f"{sha256_file(archive_path)}  {archive_path.name}\n")

    index = archive_root / "INDEX.md"
    with index.open("a") as f:
        f.write(
            f"\n## {local_dir.name}\n\n"
            f"- Archive: `{archive_path.name}`\n"
            f"- Checksum: `{checksum_path.name}`\n"
            f"- Summary: `../{local_dir.name}/live_alignment_summary.md`\n"
            f"- Raw gzip: `../{local_dir.name}/raw_market_data/`\n"
            f"- Created at: {datetime.now(timezone.utc).isoformat()}\n"
        )
    return archive_path


def write_summary(
    *,
    local_dir: Path,
    run_id: str,
    requested_run_id: str,
    audit_csv: Path,
    window: LiveWindow,
    latency_stats: dict[str, Any],
    manifest_path: Path,
    initial_state: LiveInitialState | None,
    results: dict[str, Any],
    archive_path: Path | None,
) -> Path:
    lines = [
        f"# Live alignment summary: {run_id}",
        "",
        "## Inputs",
        "",
        f"- Requested run ID: `{requested_run_id}`",
        f"- Live audit run ID: `{run_id}`",
        f"- Audit CSV: `{audit_csv}`",
        f"- Live rows: `{window.rows}`",
        f"- First ts_local: `{window.first_ts_local}`",
        f"- Last ts_local: `{window.last_ts_local}`",
        f"- Start/end day UTC: `{window.start_day}` / `{window.end_day}`",
        f"- Live raw manifest: `{manifest_path}`",
        f"- Initial position source: `{initial_state.source_field if initial_state else ''}`",
        f"- Initial position: `{initial_state.position if initial_state else 0.0}`",
        "",
        "## Latency Model",
        "",
    ]
    for key, value in latency_stats.items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Backtest Alignment", ""])
    for mode in ("normal", "audit_replay"):
        result = results.get(mode, {})
        lines.append(f"### {mode}")
        if result.get("skipped"):
            lines.append("")
            lines.append("- Status: skipped")
            lines.append("")
            continue
        report_path = result.get("alignment_report")
        lines.append("")
        lines.append(f"- Audit CSV: `{result.get('audit_csv', '')}`")
        lines.append(f"- Rows: `{result.get('rows', '')}`")
        lines.append(f"- Audit rows written: `{result.get('audit_rows', '')}`")
        lines.append(f"- Alignment report: `{report_path}`")
        lines.append(
            f"- Slice ts_local start/end: `{result.get('slice_ts_local_start', '')}` / "
            f"`{result.get('slice_ts_local_end', '')}`"
        )
        if mode == "audit_replay":
            lag_gate = result.get("audit_replay_lag_gate", {})
            lines.append(f"- Decision first ts_local: `{result.get('decision_first_ts_local', '')}`")
            lines.append(f"- First feed ts_local: `{result.get('first_feed_ts_local', '')}`")
            lines.append(f"- Audit replay prewarm ms: `{result.get('audit_replay_prewarm_ms', '')}`")
            lines.append(
                f"- Strict lag gate passed/breaches: "
                f"`{lag_gate.get('passed', '')}` / `{lag_gate.get('breach_count', '')}`"
            )
        lines.append(
            f"- Audit replay consumed/scheduled: "
            f"`{result.get('audit_replay_consumed_count', 0)}` / "
            f"`{result.get('audit_replay_scheduled_count', 0)}`"
        )
        if report_path and Path(str(report_path)).exists():
            report = json.loads(Path(str(report_path)).read_text())
            lines.append(f"- Action match rate: `{report['alignment']['action_match_rate']}`")
            lines.append(
                f"- Reject reason match rate: `{report['alignment']['reject_reason_match_rate']}`"
            )
            lines.append(f"- BT/live latency drop: `{report['bt_summary']['drop_latency_rate']}` / `{report['live_summary']['drop_latency_rate']}`")
            lines.append(f"- BT/live API drop: `{report['bt_summary']['drop_api_rate']}` / `{report['live_summary']['drop_api_rate']}`")
            lines.append(f"- MAE: `{report['alignment']['mae']}`")
        lines.append("")

    lines.extend(["## Archive", ""])
    if archive_path:
        lines.append(f"- Archive: `{archive_path}`")
    else:
        lines.append("- Archive: skipped")
    lines.append("")

    summary_path = local_dir / "live_alignment_summary.md"
    summary_path.write_text("\n".join(lines))
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch, replay, compare, and archive a live Binance tick-mm run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--remote-host", default="admin@awsserver1")
    parser.add_argument("--remote-root", default="/home/admin/hft_live")
    parser.add_argument("--local-root", default=str(PROJECT_ROOT / "local_live_analysis"))
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--config", default=None, help="Optional local live config path; defaults to fetched config_live.toml")
    parser.add_argument("--skip-fetch", action="store_true", default=False)
    parser.add_argument("--skip-backtest", action="store_true", default=False)
    parser.add_argument("--skip-archive", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--raw-opt", default="")
    parser.add_argument("--raw-buffer-size", type=int, default=100_000_000)
    parser.add_argument("--audit-replay-prewarm-ms", type=float, default=3000.0)
    parser.add_argument("--latency-mode", choices=["synthetic", "observed"], default="observed")
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
    local_root = _expand(args.local_root)
    local_dir = local_root / args.run_id

    if not args.skip_fetch:
        fetch_remote_artifacts(
            run_id=args.run_id,
            local_dir=local_dir,
            remote_host=args.remote_host,
            remote_root=args.remote_root,
            dry_run=bool(args.dry_run),
        )
        if args.dry_run:
            return

    audit_csv = find_audit_csv(local_dir, args.run_id)
    identity = read_live_audit_run_id(audit_csv, args.run_id)
    initial_state = read_live_initial_state(audit_csv)
    raw_files = sorted((local_dir / "raw_market_data").glob("*.gz"))
    if not raw_files:
        raise FileNotFoundError(f"raw gzip not found: {local_dir / 'raw_market_data'}")

    config_path = _expand(args.config) if args.config else local_dir / "config_live.toml"
    if not config_path.exists():
        copied = _copy_first_existing(
            [PROJECT_ROOT / "examples/binance_tick_mm/config.example.toml"],
            local_dir / "config_live.toml",
        )
        if copied is None:
            raise FileNotFoundError(f"config not found: {config_path}")
        config_path = copied
    base_cfg = _read_toml(config_path)

    window = read_live_window(audit_csv)
    latency_npz = local_dir / "live_order_latency.npz"
    latency_stats_path = local_dir / "live_order_latency_stats.json"
    latency_stats = generate_latency(
        audit_csv=audit_csv,
        output_npz=latency_npz,
        output_stats=latency_stats_path,
        latency_mode=str(args.latency_mode),
        entry_min_ms=float(args.entry_ms_min),
        entry_max_ms=float(args.entry_ms_max),
        resp_min_ms=float(args.resp_ms_min),
        resp_max_ms=float(args.resp_ms_max),
        spike_prob=float(args.spike_prob),
        spike_min_ms=float(args.spike_ms_min),
        spike_max_ms=float(args.spike_ms_max),
        seed=int(args.seed),
    )

    manifest_path = convert_raw_market_data(
        raw_files=raw_files,
        out_dir=local_dir / "out/live_raw",
        symbol=str(args.symbol),
        window=window,
        buffer_size=int(args.raw_buffer_size),
        opt=str(args.raw_opt),
    )

    results = run_alignment_backtests(
        base_cfg=base_cfg,
        local_dir=local_dir,
        run_id=identity.audit_run_id,
        audit_csv=audit_csv,
        latency_npz=latency_npz,
        manifest_path=manifest_path,
        window=window,
        initial_state=initial_state,
        skip_backtest=bool(args.skip_backtest),
        audit_replay_prewarm_ms=float(args.audit_replay_prewarm_ms),
    )

    archive_path = None if args.skip_archive else (local_root / "archive" / f"{local_dir.name}.tar.gz")
    summary_path = write_summary(
        local_dir=local_dir,
        run_id=identity.audit_run_id,
        requested_run_id=identity.requested_run_id,
        audit_csv=audit_csv,
        window=window,
        latency_stats=latency_stats,
        manifest_path=manifest_path,
        initial_state=initial_state,
        results=results,
        archive_path=archive_path,
    )

    if not args.skip_archive:
        archive_path = archive_run(local_dir, local_root / "archive")

    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "local_dir": str(local_dir),
                "audit_csv": str(audit_csv),
                "manifest": str(manifest_path),
                "summary": str(summary_path),
                "archive": str(archive_path) if archive_path else "",
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
