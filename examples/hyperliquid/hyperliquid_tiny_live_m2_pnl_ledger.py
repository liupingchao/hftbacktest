#!/usr/bin/env python3
"""M2A real PnL ledger and reconciliation gate.

This runner is intentionally no-network. It turns already-pulled Hyperliquid
tiny-live artifacts, plus optional fill/economics/inventory fixtures, into a
fail-closed PnL ledger. Missing fills or missing settlement evidence must never
be interpreted as realized PnL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0618T008"
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_m1_canary_loop_0618T007"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_m2_pnl_ledger_0618T008"
READY_RECOMMENDATION = "hyperliquid_tiny_live_m2_pnl_ledger_ready_for_qa"
BLOCKED_RECOMMENDATION = "hyperliquid_tiny_live_m2_pnl_ledger_blocked"
SCHEMA_VERSION = "hyperliquid_tiny_live_m2_pnl_ledger_v1"

ADDRESS_RE = re.compile(r"0x[a-fA-F0-9]{40}")
HEX_32_RE = re.compile(r"0x[a-fA-F0-9]{32}")
HEX_64_RE = re.compile(r"0x[a-fA-F0-9]{64}")
REDACT_KEYS = {
    "account",
    "address",
    "api_key",
    "api_secret",
    "authorization",
    "cloid",
    "nonce",
    "oid",
    "private_key",
    "raw_signature",
    "secret",
    "signature",
    "user",
    "wallet",
}


class LedgerError(RuntimeError):
    """Raised when the ledger input cannot be parsed."""


@dataclass(frozen=True)
class FillRecord:
    source_window: str
    fill_id: str
    side: str
    qty_btc: float
    price_usdc: float
    fee_usdc: float
    rebate_usdc: float
    liquidity: str
    intent_price_usdc: float | None
    mark_price_usdc: float | None

    @property
    def signed_qty_btc(self) -> float:
        return self.qty_btc if self.side == "buy" else -self.qty_btc

    @property
    def gross_pnl_usdc(self) -> float:
        if self.mark_price_usdc is None:
            return 0.0
        if self.side == "buy":
            return (self.mark_price_usdc - self.price_usdc) * self.qty_btc
        return (self.price_usdc - self.mark_price_usdc) * self.qty_btc

    @property
    def net_fee_usdc(self) -> float:
        return self.fee_usdc - self.rebate_usdc

    @property
    def net_pnl_usdc(self) -> float:
        return self.gross_pnl_usdc - self.net_fee_usdc

    @property
    def slippage_usdc(self) -> float:
        if self.intent_price_usdc is None:
            return 0.0
        if self.side == "buy":
            return (self.price_usdc - self.intent_price_usdc) * self.qty_btc
        return (self.intent_price_usdc - self.price_usdc) * self.qty_btc


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "<redacted>" if str(key).lower() in REDACT_KEYS else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        value = HEX_64_RE.sub("<redacted_hex64>", value)
        value = ADDRESS_RE.sub("<redacted_address>", value)
        value = HEX_32_RE.sub("<redacted_hex32>", value)
        return value
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redact(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: redact(row.get(field, "")) for field in fieldnames})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LedgerError(f"missing_json:{path}") from exc


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def parse_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def discover_windows(input_root: Path) -> list[Path]:
    if not input_root.exists():
        return []
    windows = sorted(path for path in input_root.glob("window_*/pulled_back_awsserver1") if path.is_dir())
    if windows:
        return windows
    if (input_root / "executor_manifest.json").exists():
        return [input_root]
    return []


def intent_by_window(window_dir: Path) -> dict[str, Any]:
    rows = read_csv_rows(window_dir / "order_intent_audit.csv")
    if not rows:
        return {}
    row = rows[0]
    return {
        "side": row.get("side", ""),
        "size_btc": parse_float(row.get("size_btc")),
        "limit_px": parse_float(row.get("limit_px")),
        "time_in_force": row.get("time_in_force", ""),
        "endpoint_called": row.get("endpoint_called", ""),
    }


def source_completeness_for_window(window_dir: Path) -> dict[str, Any]:
    manifest = read_json(window_dir / "executor_manifest.json")
    private_response = read_json(window_dir / "private_order_response_audit.json")
    private_preflight = read_json(window_dir / "private_preflight_summary.json")
    cancel_proof = read_json(window_dir / "cancel_shutdown_proof.json")
    live_fill_rows = read_csv_rows(window_dir / "live_fill_ledger.csv")
    status_types = manifest.get("order_status_types", [])
    fill_count_before = (
        private_preflight.get("preflight_summary", {})
        .get("client_preflight", {})
        .get("user_fill_count", private_preflight.get("preflight_summary", {}).get("user_fill_count_before", 0))
    )
    filled_rows = [
        row for row in private_response.get("order_status_rows", [])
        if isinstance(row, dict) and row.get("status_type") in {"filled", "partialFill", "filledResting", "filledCrossed"}
    ]
    fills_available = len(filled_rows) > 0 or len(live_fill_rows) > 0
    economics_available = any(
        row.get("fee_usdc") not in {"", None} or row.get("rebate_usdc") not in {"", None}
        for row in live_fill_rows
    )
    return {
        "window": window_dir.parent.name if window_dir.name == "pulled_back_awsserver1" else window_dir.name,
        "artifact_dir": str(window_dir),
        "order_submission_attempted": bool(manifest.get("order_submission_attempted")),
        "order_status_types": ",".join(str(item) for item in status_types),
        "private_endpoint_called": bool(manifest.get("private_endpoint_called")),
        "real_order_endpoint_called": bool(manifest.get("real_order_endpoint_called")),
        "real_cancel_endpoint_called": bool(manifest.get("real_cancel_endpoint_called")),
        "shutdown_proof_status": manifest.get("shutdown_proof_status", ""),
        "final_open_orders_count": len(cancel_proof.get("final_open_orders", [])),
        "user_fill_count_before": fill_count_before,
        "fill_status_rows_after_order": max(len(filled_rows), len(live_fill_rows)),
        "fills_available": fills_available,
        "economics_settlement_available": economics_available,
        "inventory_snapshot_available": private_preflight.get("preflight_summary", {}).get("asset_position_count_before", 0) not in ("", None),
        "realized_pnl_proof_status": "candidate_live_fill_requires_ledger" if fills_available else "unavailable_no_fill_or_settlement",
        "fail_closed_reason": "" if fills_available else "m1_canary_rested_then_cancelled_without_fill_economics_or_inventory_transition",
    }


def load_fixture_fills(path: Path | None) -> list[FillRecord]:
    if path is None:
        return []
    rows = read_csv_rows(path)
    fills: list[FillRecord] = []
    for idx, row in enumerate(rows, start=1):
        side = row.get("side", "").lower()
        if side not in {"buy", "sell"}:
            raise LedgerError(f"fixture_fill_{idx}_bad_side:{side}")
        qty = parse_float(row.get("qty_btc"))
        price = parse_float(row.get("price_usdc"))
        if qty <= 0 or price <= 0:
            raise LedgerError(f"fixture_fill_{idx}_bad_qty_or_price")
        fills.append(
            FillRecord(
                source_window=row.get("source_window", "fixture"),
                fill_id=row.get("fill_id", f"fixture_{idx}"),
                side=side,
                qty_btc=qty,
                price_usdc=price,
                fee_usdc=parse_float(row.get("fee_usdc")),
                rebate_usdc=parse_float(row.get("rebate_usdc")),
                liquidity=row.get("liquidity", "unknown").lower(),
                intent_price_usdc=parse_float(row.get("intent_price_usdc"), default=0.0) or None,
                mark_price_usdc=parse_float(row.get("mark_price_usdc"), default=0.0) or None,
            )
        )
    return fills


def summarize_fills(fills: list[FillRecord]) -> dict[str, Any]:
    gross = sum(fill.gross_pnl_usdc for fill in fills)
    fees = sum(fill.fee_usdc for fill in fills)
    rebates = sum(fill.rebate_usdc for fill in fills)
    net_fee = sum(fill.net_fee_usdc for fill in fills)
    net = sum(fill.net_pnl_usdc for fill in fills)
    slippage = sum(fill.slippage_usdc for fill in fills)
    inventory_delta = sum(fill.signed_qty_btc for fill in fills)
    notional = sum(fill.qty_btc * fill.price_usdc for fill in fills)
    maker_count = sum(1 for fill in fills if fill.liquidity == "maker")
    missing_mark = sum(1 for fill in fills if fill.mark_price_usdc is None)
    return {
        "fill_count": len(fills),
        "maker_fill_count": maker_count,
        "gross_pnl_usdc": round(gross, 8),
        "fee_usdc": round(fees, 8),
        "rebate_usdc": round(rebates, 8),
        "net_fee_usdc": round(net_fee, 8),
        "net_pnl_usdc": round(net, 8),
        "slippage_usdc": round(slippage, 8),
        "inventory_delta_btc": round(inventory_delta, 10),
        "filled_notional_usdc": round(notional, 8),
        "missing_mark_count": missing_mark,
    }


def fill_rows(fills: list[FillRecord]) -> list[dict[str, Any]]:
    rows = []
    for fill in fills:
        rows.append(
            {
                "source_window": fill.source_window,
                "fill_id": fill.fill_id,
                "side": fill.side,
                "qty_btc": fill.qty_btc,
                "price_usdc": fill.price_usdc,
                "intent_price_usdc": fill.intent_price_usdc if fill.intent_price_usdc is not None else "",
                "mark_price_usdc": fill.mark_price_usdc if fill.mark_price_usdc is not None else "",
                "liquidity": fill.liquidity,
                "gross_pnl_usdc": round(fill.gross_pnl_usdc, 8),
                "fee_usdc": fill.fee_usdc,
                "rebate_usdc": fill.rebate_usdc,
                "net_fee_usdc": round(fill.net_fee_usdc, 8),
                "net_pnl_usdc": round(fill.net_pnl_usdc, 8),
                "slippage_usdc": round(fill.slippage_usdc, 8),
                "signed_qty_btc": round(fill.signed_qty_btc, 10),
            }
        )
    return rows


def overclaim_rows(*, source_rows: list[dict[str, Any]], fills: list[FillRecord]) -> list[dict[str, Any]]:
    no_live_fills = all(not row.get("fills_available") for row in source_rows)
    return [
        {
            "claim": "realized_pnl_from_m1_canary",
            "status": "fail_closed" if no_live_fills else "requires_reconciliation",
            "reason": "m1_artifacts_have_no_fill_or_economics_settlement" if no_live_fills else "fills_present_must_reconcile",
        },
        {
            "claim": "fee_rebate_proof",
            "status": "pass" if fills else "fail_closed",
            "reason": "" if fills else "missing_economics_settlement_records",
        },
        {
            "claim": "inventory_lifecycle_proof",
            "status": "pass" if fills else "fail_closed",
            "reason": "" if fills else "missing_account_inventory_transition",
        },
        {
            "claim": "stable_pnl_or_maker_viability",
            "status": "fail_closed",
            "reason": "M2A ledger mechanics do not prove stability or maker viability",
        },
    ]


def artifact_nonempty_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(output_dir)),
                    "size_bytes": path.stat().st_size,
                    "status": "pass" if path.stat().st_size > 0 else "fail",
                }
            )
    return rows


def build_sha256_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.csv":
            rows.append({"artifact": str(path.relative_to(output_dir)), "sha256": sha256(path)})
    write_csv(output_dir / "sha256_manifest.csv", rows, ["artifact", "sha256"])


def write_fixture(path: Path) -> None:
    rows = [
        {
            "source_window": "fixture_window_1",
            "fill_id": "fixture_fill_1",
            "side": "buy",
            "qty_btc": "0.01",
            "price_usdc": "62634.0",
            "intent_price_usdc": "62634.0",
            "mark_price_usdc": "62660.0",
            "fee_usdc": "0.125268",
            "rebate_usdc": "0.0",
            "liquidity": "maker",
        }
    ]
    write_csv(
        path,
        rows,
        [
            "source_window",
            "fill_id",
            "side",
            "qty_btc",
            "price_usdc",
            "intent_price_usdc",
            "mark_price_usdc",
            "fee_usdc",
            "rebate_usdc",
            "liquidity",
        ],
    )


def run_ledger(
    *,
    input_root: Path,
    output_dir: Path,
    fixture_fills: Path | None = None,
    fill_ledger: Path | None = None,
    fill_source_kind: str = "fixture",
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = discover_windows(input_root)
    source_rows = [source_completeness_for_window(window) for window in windows]
    fill_path = fill_ledger or fixture_fills
    if fill_ledger is not None and fixture_fills is not None:
        raise LedgerError("use either fixture_fills or fill_ledger, not both")
    if fill_source_kind not in {"fixture", "live_pulled_back"}:
        raise LedgerError(f"unsupported_fill_source_kind:{fill_source_kind}")
    fills = load_fixture_fills(fill_path)
    summary = summarize_fills(fills)
    has_fixture = fill_path is not None and fill_source_kind == "fixture"
    source_ok = bool(source_rows)
    if not source_ok:
        final_recommendation = BLOCKED_RECOMMENDATION
        blocking_reasons = ["no_input_windows_found"]
    else:
        final_recommendation = READY_RECOMMENDATION
        blocking_reasons = []

    live_realized_pnl_proof = (
        fill_source_kind == "live_pulled_back"
        and any(row.get("fills_available") for row in source_rows)
        and bool(fills)
    )
    if not live_realized_pnl_proof:
        proof_status = "fail_closed_no_realized_live_pnl"
    elif summary["missing_mark_count"]:
        proof_status = "fail_closed_missing_markout"
    else:
        proof_status = "pass_fixture_only" if has_fixture else "pass"

    write_csv(
        output_dir / "source_completeness_matrix.csv",
        source_rows,
        [
            "window",
            "artifact_dir",
            "order_submission_attempted",
            "order_status_types",
            "private_endpoint_called",
            "real_order_endpoint_called",
            "real_cancel_endpoint_called",
            "shutdown_proof_status",
            "final_open_orders_count",
            "user_fill_count_before",
            "fill_status_rows_after_order",
            "fills_available",
            "economics_settlement_available",
            "inventory_snapshot_available",
            "realized_pnl_proof_status",
            "fail_closed_reason",
        ],
    )
    write_csv(
        output_dir / "fill_ledger.csv",
        fill_rows(fills),
        [
            "source_window",
            "fill_id",
            "side",
            "qty_btc",
            "price_usdc",
            "intent_price_usdc",
            "mark_price_usdc",
            "liquidity",
            "gross_pnl_usdc",
            "fee_usdc",
            "rebate_usdc",
            "net_fee_usdc",
            "net_pnl_usdc",
            "slippage_usdc",
            "signed_qty_btc",
        ],
    )
    write_json(output_dir / "pnl_summary.json", summary)
    write_csv(output_dir / "overclaim_gate_matrix.csv", overclaim_rows(source_rows=source_rows, fills=fills), ["claim", "status", "reason"])
    write_json(
        output_dir / "replay_live_comparison_limits.json",
        {
            "optimistic_proxy_is_realized_pnl": False,
            "m1_canary_is_realized_pnl": False,
            "fixture_rows_are_live_pnl_proof": False,
            "fill_source_kind": fill_source_kind,
            "allowed_use": "M2B ledger schema and reconciliation gate",
            "forbidden_use": "stable PnL, maker viability, promotion, scale-up, or default-on readiness",
        },
    )
    write_json(
        output_dir / "m2_pnl_ledger_manifest.json",
        {
            "task_id": TASK_ID,
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "git_commit": git_commit(),
            "final_recommendation": final_recommendation,
            "blocking_reasons": blocking_reasons,
            "input_root": str(input_root),
            "fixture_fills": str(fixture_fills) if fixture_fills else "",
            "fill_ledger": str(fill_ledger) if fill_ledger else "",
            "fill_source_kind": fill_source_kind,
            "windows_found": len(windows),
            "real_orders_placed": False,
            "private_endpoint_called": False,
            "account_endpoint_called": False,
            "credentials_read": False,
            "live_realized_pnl_proof": live_realized_pnl_proof,
            "realized_pnl_proof_status": proof_status,
            "m1_no_fill_fail_closed": not live_realized_pnl_proof,
            "ledger_summary": summary,
            "output_files": {
                "source_completeness_matrix": str(output_dir / "source_completeness_matrix.csv"),
                "fill_ledger": str(output_dir / "fill_ledger.csv"),
                "pnl_summary": str(output_dir / "pnl_summary.json"),
                "overclaim_gate_matrix": str(output_dir / "overclaim_gate_matrix.csv"),
                "replay_live_comparison_limits": str(output_dir / "replay_live_comparison_limits.json"),
            },
        },
    )
    readme = [
        "# Hyperliquid M2A PnL Ledger",
        "",
        f"Final recommendation: `{final_recommendation}`",
        "",
        "This no-network artifact set establishes the M2 ledger schema and fail-closed reconciliation gates.",
        "",
        "M1 canary artifacts are not realized PnL proof when fills, fee settlement, and inventory transitions are absent.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    write_csv(output_dir / "artifact_nonempty_check.csv", artifact_nonempty_rows(output_dir), ["path", "size_bytes", "status"])
    build_sha256_manifest(output_dir)
    return read_json(output_dir / "m2_pnl_ledger_manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fixture-fills", type=Path, default=None)
    parser.add_argument("--fill-ledger", type=Path, default=None)
    parser.add_argument("--fill-source-kind", choices=["fixture", "live_pulled_back"], default="fixture")
    parser.add_argument("--write-sample-fixture", type=Path, default=None)
    args = parser.parse_args()
    if args.write_sample_fixture:
        write_fixture(args.write_sample_fixture)
        print(json.dumps({"sample_fixture": str(args.write_sample_fixture)}, indent=2, sort_keys=True))
        return 0
    manifest = run_ledger(
        input_root=args.input_root,
        output_dir=args.output_dir,
        fixture_fills=args.fixture_fills,
        fill_ledger=args.fill_ledger,
        fill_source_kind=args.fill_source_kind,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
