#!/usr/bin/env python3
"""Preflight checks and deployment manifest for Binance maker live runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import tomllib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class PreflightError(RuntimeError):
    """Raised when a live run should fail before any live process starts."""


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise PreflightError(f"{label} not found: {resolved}")
    return resolved


def read_toml(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise PreflightError(f"{label} is not valid TOML: {path}: {exc}") from exc


def run_git(project_root: Path, args: list[str], *, allow_fail: bool = False) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=project_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0 and not allow_fail:
        message = result.stderr.strip() or result.stdout.strip()
        raise PreflightError(f"git {' '.join(args)} failed: {message}")
    return result.stdout.strip()


def git_info(project_root: Path) -> dict[str, Any]:
    commit = run_git(project_root, ["rev-parse", "HEAD"])
    short_commit = run_git(project_root, ["rev-parse", "--short", "HEAD"])
    branch = run_git(project_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    status_short = run_git(project_root, ["status", "--short"], allow_fail=True)
    return {
        "commit": commit,
        "short_commit": short_commit,
        "branch": branch,
        "dirty": bool(status_short),
        "status_short": status_short.splitlines(),
    }


def load_strategy_core(example_dir: Path) -> Any:
    example_dir_str = str(example_dir)
    if example_dir_str not in sys.path:
        sys.path.insert(0, example_dir_str)
    try:
        import strategy_core  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - error message is the behavior.
        raise PreflightError(f"failed to import strategy_core from {example_dir}: {exc}") from exc
    return strategy_core


def duplicate_fields(fields: list[str]) -> list[str]:
    counts = Counter(fields)
    return sorted(field for field, count in counts.items() if count > 1)


def validate_audit_row_keys(audit_fields: list[str], row_keys: list[str], *, label: str) -> dict[str, Any]:
    duplicates = duplicate_fields(audit_fields)
    field_set = set(audit_fields)
    row_key_set = set(row_keys)
    missing = sorted(field_set - row_key_set)
    extra = sorted(row_key_set - field_set)
    if duplicates or missing or extra:
        raise PreflightError(
            f"{label} is incompatible with AUDIT_FIELDS: "
            f"duplicates={duplicates}, missing={missing}, extra={extra}"
        )
    return {
        "field_count": len(audit_fields),
        "row_key_count": len(row_keys),
        "duplicates": duplicates,
        "missing": missing,
        "extra": extra,
    }


def build_sample_decision_row(strategy_core: Any, symbol: str) -> dict[str, Any]:
    return strategy_core.build_audit_row(
        run_id="preflight",
        symbol=symbol,
        strategy_seq=1,
        ts_local=1_000_000_000,
        ts_exch=999_000_000,
        action_order_id="",
        action_name="keep",
        planned_order_id="",
        planned_action="keep",
        throttle_reason="",
        reject_reason="",
        req_ts=0,
        exch_ts=0,
        resp_ts=0,
        entry_latency_ns=0,
        resp_latency_ns=0,
        predicted_entry_ns=0,
        best_bid=100.0,
        best_ask=100.1,
        mid=100.05,
        fair=100.05,
        reservation=100.05,
        half_spread=0.05,
        position=0.0,
        auditlatency_ms=0.0,
        dropped_by_latency=False,
        dropped_by_api_limit=False,
        pos_limit=False,
        impact_cost_val=0.0,
        spread_bps=1.0,
        vol_bps=0.0,
        inventory_score=0.0,
        feed_latency_ns=0,
        latency_signal_ns=0,
        bid_size=1.0,
        ask_size=1.0,
        bid_top5_ticks="1000|999|998|997|996",
        bid_top5_qtys="1.0|0.0|0.0|0.0|0.0",
        ask_top5_ticks="1001|1002|1003|1004|1005",
        ask_top5_qtys="1.0|0.0|0.0|0.0|0.0",
        market_view_source="preflight",
        top5_source="preflight",
        market_overlay_source="",
        top5_overlay_source="",
        book_view_ts_local=1_000_000_000,
        book_view_ts_exch=999_000_000,
        book_view_feed_latency_ns=1_000_000,
        book_view_stale_ms=1.0,
        top5_depth_best_bid_tick=1000,
        top5_depth_best_ask_tick=1001,
        greek_values=strategy_core.GreekValues(0.0, 0.0, 0.0, 0.0),
        greek_adjustment=0.0,
        target_bid_tick=1000,
        target_ask_tick=1001,
        working_bid_tick=-1,
        working_ask_tick=-1,
        working_buy_order_id="",
        working_sell_order_id="",
        extra_order_ids="",
        extra_order_sides="",
        extra_order_price_ticks="",
    )


def build_sample_lifecycle_row(strategy_core: Any, symbol: str) -> dict[str, Any]:
    return strategy_core.build_lifecycle_event_row(
        run_id="preflight",
        symbol=symbol,
        strategy_seq=1,
        event_seq=1,
        event_type="lifecycle_preflight",
        event_source="preflight",
        ts_local=1_000_000_000,
        ts_exch=999_000_000,
        best_bid=100.0,
        best_ask=100.1,
        mid=100.05,
        position=0.0,
    )


def schema_compatibility(example_dir: Path, symbol: str) -> dict[str, Any]:
    strategy_core = load_strategy_core(example_dir)
    audit_fields = list(strategy_core.AUDIT_FIELDS)
    decision_row = build_sample_decision_row(strategy_core, symbol)
    lifecycle_row = build_sample_lifecycle_row(strategy_core, symbol)
    decision = validate_audit_row_keys(
        audit_fields,
        list(decision_row.keys()),
        label="strategy_core.build_audit_row",
    )
    lifecycle = validate_audit_row_keys(
        audit_fields,
        list(lifecycle_row.keys()),
        label="strategy_core.build_lifecycle_event_row",
    )
    return {
        "passed": True,
        "audit_field_count": len(audit_fields),
        "decision_row": decision,
        "lifecycle_row": lifecycle,
    }


def build_manifest(
    *,
    project_root: Path,
    config: Path,
    connector_config: Path,
    symbol: str,
    data_dir: Path,
    run_dir: Path,
    manifest_out: Path,
    start_marker_out: Path | None,
    stop_marker_out: Path | None,
) -> dict[str, Any]:
    project_root = project_root.expanduser().resolve()
    example_dir = project_root / "examples" / "binance_tick_mm"
    deploy_dir = example_dir / "deploy"
    config = require_file(config, "config")
    connector_config = require_file(connector_config, "connector config")
    config_doc = read_toml(config, "config")
    read_toml(connector_config, "connector config")

    code_files = {
        "audit_schema": example_dir / "audit_schema.py",
        "strategy_core": example_dir / "strategy_core.py",
        "live_tick_mm": example_dir / "live_tick_mm.py",
        "run_live": deploy_dir / "run_live.sh",
        "preflight_live_run": deploy_dir / "preflight_live_run.py",
    }
    resolved_code_files = {name: require_file(path, name) for name, path in code_files.items()}

    compatibility = schema_compatibility(example_dir, symbol)
    hashes = {
        "config": sha256_file(config),
        "connector_config": sha256_file(connector_config),
        **{name: sha256_file(path) for name, path in resolved_code_files.items()},
    }

    live_cfg = config_doc.get("live", {}) if isinstance(config_doc.get("live", {}), dict) else {}
    paths_cfg = config_doc.get("paths", {}) if isinstance(config_doc.get("paths", {}), dict) else {}

    return {
        "manifest_version": 1,
        "generated_at_utc": utc_now_iso(),
        "host": socket.gethostname(),
        "project_root": str(project_root),
        "example_dir": str(example_dir),
        "git": git_info(project_root),
        "inputs": {
            "symbol": symbol,
            "config": str(config),
            "connector_config": str(connector_config),
            "data_dir": str(data_dir.expanduser().resolve()),
            "run_dir": str(run_dir.expanduser().resolve()),
            "manifest_out": str(manifest_out.expanduser().resolve()),
        },
        "configured_outputs": {
            "live_audit_csv": live_cfg.get("audit_csv", ""),
            "output_root": paths_cfg.get("output_root", ""),
        },
        "markers": {
            "start_marker": str(start_marker_out.expanduser().resolve()) if start_marker_out else "",
            "stop_marker": str(stop_marker_out.expanduser().resolve()) if stop_marker_out else "",
        },
        "hashes": hashes,
        "code_files": {name: str(path) for name, path in resolved_code_files.items()},
        "compatibility": compatibility,
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
        },
        "decision": "preflight_passed",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_start_marker(path: Path, manifest: dict[str, Any]) -> None:
    marker = {
        "event": "preflight_passed_before_tmux_start",
        "generated_at_utc": utc_now_iso(),
        "host": manifest["host"],
        "git": manifest["git"],
        "manifest": manifest["inputs"]["manifest_out"],
        "symbol": manifest["inputs"]["symbol"],
    }
    write_json(path, marker)


def write_stop_marker(path: Path, manifest_path: Path, exit_code: int) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = {
        "event": "live_bot_process_exited",
        "generated_at_utc": utc_now_iso(),
        "host": socket.gethostname(),
        "manifest": str(manifest_path),
        "exit_code": int(exit_code),
        "git": manifest.get("git", {}),
        "symbol": manifest.get("inputs", {}).get("symbol", ""),
    }
    write_json(path, marker)
    return marker


def run_preflight(
    *,
    project_root: Path,
    config: Path,
    connector_config: Path,
    symbol: str,
    data_dir: Path,
    run_dir: Path,
    manifest_out: Path,
    start_marker_out: Path | None = None,
    stop_marker_out: Path | None = None,
    write_start: bool = True,
) -> dict[str, Any]:
    manifest = build_manifest(
        project_root=project_root,
        config=config,
        connector_config=connector_config,
        symbol=symbol,
        data_dir=data_dir,
        run_dir=run_dir,
        manifest_out=manifest_out,
        start_marker_out=start_marker_out,
        stop_marker_out=stop_marker_out,
    )
    write_json(manifest_out, manifest)
    if write_start and start_marker_out is not None:
        write_start_marker(start_marker_out, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Binance maker live deployment compatibility and write a run manifest"
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--connector-config", type=Path)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--start-marker-out", type=Path, default=None)
    parser.add_argument("--stop-marker-out", type=Path, default=None)
    parser.add_argument("--manifest-in", type=Path, default=None)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument(
        "--no-start-marker",
        action="store_true",
        help="Do not write the pre-tmux start marker after preflight passes",
    )
    parser.add_argument(
        "--write-stop-marker-only",
        action="store_true",
        help="Write only the stop marker from an existing manifest",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_stop_marker_only:
        if args.manifest_in is None or args.stop_marker_out is None:
            print(
                "preflight error: --write-stop-marker-only requires "
                "--manifest-in and --stop-marker-out",
                file=sys.stderr,
            )
            return 2
        marker = write_stop_marker(args.stop_marker_out, args.manifest_in, args.exit_code)
        print(f"stop marker written: {args.stop_marker_out} exit_code={marker['exit_code']}")
        return 0

    missing_args = [
        name
        for name in [
            "project_root",
            "config",
            "connector_config",
            "data_dir",
            "run_dir",
            "manifest_out",
        ]
        if getattr(args, name) is None
    ]
    if missing_args:
        print(f"preflight error: missing required args: {', '.join(missing_args)}", file=sys.stderr)
        return 2

    try:
        manifest = run_preflight(
            project_root=args.project_root,
            config=args.config,
            connector_config=args.connector_config,
            symbol=args.symbol,
            data_dir=args.data_dir,
            run_dir=args.run_dir,
            manifest_out=args.manifest_out,
            start_marker_out=args.start_marker_out,
            stop_marker_out=args.stop_marker_out,
            write_start=not args.no_start_marker,
        )
    except PreflightError as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 1

    print(
        "preflight passed: "
        f"commit={manifest['git']['short_commit']} "
        f"schema_fields={manifest['compatibility']['audit_field_count']} "
        f"manifest={manifest['inputs']['manifest_out']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
