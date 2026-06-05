#!/usr/bin/env python3
"""Local no-order dry-run for shutdown final proof observability."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hftbacktest import CANCELED, NEW

from audit_schema import AUDIT_FIELDS
from live_tick_mm import (
    SHUTDOWN_CANCEL_ACK_TIMEOUT_NS,
    _append_shutdown_final_proof_audit_tail,
    cancel_working_orders_for_shutdown,
)
from strategy_core import WorkingOrders


TASK_ID = "0605T004"
RUN_ID = "shutdown-proof-dry-run-0605T004"
SYMBOL = "BTCUSDT"


@dataclass(frozen=True)
class DryRunScenario:
    name: str
    wait_result: int
    local_status: int | None
    rest_open_order_ids: tuple[int, ...]
    rest_fails: bool = False
    use_rest_client: bool = True
    expected_final_proof_level: str = ""


class _FakeOrder:
    def __init__(self, order_id: int, *, status: int = NEW) -> None:
        self.order_id = int(order_id)
        self.status = int(status)
        self.cancellable = True
        self.price_tick = 770000
        self.side = 1
        self.qty = 0.001
        self.leaves_qty = 0.001
        self.exec_qty = 0.0
        self.req = 1


class _FakeOrderValues:
    def __init__(self, orders: list[_FakeOrder]) -> None:
        self._orders = orders
        self._index = 0

    def has_next(self) -> bool:
        return self._index < len(self._orders)

    def get(self) -> _FakeOrder:
        order = self._orders[self._index]
        self._index += 1
        return order


class _FakeOrderDict:
    def __init__(self, orders: list[_FakeOrder]) -> None:
        self._orders = orders

    def values(self) -> _FakeOrderValues:
        return _FakeOrderValues(self._orders)


class _FakeHbt:
    def __init__(self, scenario: DryRunScenario, order_id: int) -> None:
        self.scenario = scenario
        self.order_id = int(order_id)
        self.cancel_calls: list[tuple[int, int, bool]] = []
        self.wait_calls: list[tuple[int, int, int]] = []
        self.order_snapshot_calls: list[int] = []

    def cancel(self, asset_no: int, order_id: int, wait: bool) -> None:
        self.cancel_calls.append((asset_no, order_id, wait))

    def wait_order_response(self, asset_no: int, order_id: int, timeout_ns: int) -> int:
        self.wait_calls.append((asset_no, order_id, timeout_ns))
        return int(self.scenario.wait_result)

    def orders(self, asset_no: int) -> _FakeOrderDict:
        self.order_snapshot_calls.append(asset_no)
        if self.scenario.local_status is None:
            return _FakeOrderDict([])
        return _FakeOrderDict([_FakeOrder(self.order_id, status=self.scenario.local_status)])


class _FakeRestClient:
    def __init__(self, scenario: DryRunScenario) -> None:
        self.scenario = scenario
        self.open_order_calls: list[str] = []

    def open_orders(self, symbol: str) -> list[dict[str, object]]:
        self.open_order_calls.append(symbol)
        if self.scenario.rest_fails:
            raise RuntimeError("dry-run open_orders failure")
        rows: list[dict[str, object]] = []
        for order_id in self.scenario.rest_open_order_ids:
            rows.append(
                {
                    "clientOrderId": f"dryrun-{order_id}",
                    "orderId": 900000 + int(order_id),
                    "side": "BUY",
                    "price": "77000.0",
                    "origQty": "0.001",
                    "executedQty": "0",
                    "status": "NEW",
                    "timeInForce": "GTX",
                    "updateTime": "1",
                }
            )
        return rows


def _scenarios() -> list[DryRunScenario]:
    return [
        DryRunScenario(
            name="local_absent_exchange_absent",
            wait_result=3,
            local_status=None,
            rest_open_order_ids=(),
            expected_final_proof_level="exchange_reconciled",
        ),
        DryRunScenario(
            name="local_absent_exchange_still_open",
            wait_result=3,
            local_status=None,
            rest_open_order_ids=(101,),
            expected_final_proof_level="exchange_still_open",
        ),
        DryRunScenario(
            name="exchange_check_failed",
            wait_result=0,
            local_status=None,
            rest_open_order_ids=(),
            rest_fails=True,
            expected_final_proof_level="local_only",
        ),
        DryRunScenario(
            name="local_active_exchange_absent",
            wait_result=3,
            local_status=NEW,
            rest_open_order_ids=(),
            expected_final_proof_level="exchange_absent_only",
        ),
        DryRunScenario(
            name="local_terminal_exchange_absent",
            wait_result=0,
            local_status=CANCELED,
            rest_open_order_ids=(),
            expected_final_proof_level="exchange_reconciled",
        ),
        DryRunScenario(
            name="no_rest_client",
            wait_result=3,
            local_status=None,
            rest_open_order_ids=(),
            use_rest_client=False,
            expected_final_proof_level="local_only",
        ),
    ]


def _working_order(order_id: int) -> WorkingOrders:
    return WorkingOrders(buy=_FakeOrder(order_id), sell=None, extras=[])


def _summary_row(scenario: DryRunScenario, order_id: int) -> dict[str, object]:
    hbt = _FakeHbt(scenario, order_id)
    rest_client = _FakeRestClient(scenario) if scenario.use_rest_client else None
    result = cancel_working_orders_for_shutdown(
        hbt,
        _working_order(order_id),
        rest_client=rest_client,
        symbol=SYMBOL,
        tick_size=0.1,
    )[0]
    return {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "scenario": scenario.name,
        "order_id": result.order_id,
        "wait_result_raw": result.wait_result_raw,
        "wait_outcome": result.wait_outcome,
        "order_response_received": int(result.order_response_received),
        "terminal_confirmed": int(result.terminal_confirmed),
        "terminal_confirmation_source": result.terminal_confirmation_source,
        "final_order_status": result.final_order_status,
        "exchange_reconciliation_checked": int(result.exchange_reconciliation_checked),
        "exchange_open_order_absent": int(result.exchange_open_order_absent),
        "exchange_confirmation_source": result.exchange_confirmation_source,
        "exchange_reconciliation_status": result.exchange_reconciliation_status,
        "final_proof_level": result.final_proof_level,
        "expected_final_proof_level": scenario.expected_final_proof_level,
        "passed": int(result.final_proof_level == scenario.expected_final_proof_level),
        "cancel_call_count": len(hbt.cancel_calls),
        "wait_call_count": len(hbt.wait_calls),
        "local_snapshot_call_count": len(hbt.order_snapshot_calls),
        "rest_open_orders_call_count": (
            len(rest_client.open_order_calls) if rest_client is not None else 0
        ),
        "boundary": "local_fake_no_order_no_network",
    }


def run_dry_run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "shutdown_proof_summary.csv"
    audit_path = output_dir / "shutdown_final_proof_audit_tail.csv"
    manifest_path = output_dir / "run_manifest.json"
    report_path = output_dir / "shutdown_proof_dry_run_report.md"

    rows = [_summary_row(scenario, 101) for scenario in _scenarios()]
    with summary_path.open("w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with audit_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
    for scenario in _scenarios():
        hbt = _FakeHbt(scenario, 101)
        rest_client = _FakeRestClient(scenario) if scenario.use_rest_client else None
        result = cancel_working_orders_for_shutdown(
            hbt,
            _working_order(101),
            rest_client=rest_client,
            symbol=SYMBOL,
            tick_size=0.1,
        )[0]
        _append_shutdown_final_proof_audit_tail(
            audit_path=audit_path,
            run_id=f"{RUN_ID}-{scenario.name}",
            symbol=SYMBOL,
            results=[result],
        )

    proof_counts: dict[str, int] = {}
    for row in rows:
        level = str(row["final_proof_level"])
        proof_counts[level] = proof_counts.get(level, 0) + 1
    all_passed = all(int(row["passed"]) == 1 for row in rows)
    manifest = {
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "classification": "local_fake_no_order_no_network_dry_run",
        "network_used": False,
        "live_started": False,
        "real_orders_used": False,
        "real_cancel_used": False,
        "real_rest_used": False,
        "scenario_count": len(rows),
        "all_scenarios_passed": all_passed,
        "final_proof_level_counts": proof_counts,
        "summary_csv": str(summary_path),
        "audit_tail_csv": str(audit_path),
        "report_md": str(report_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    report_lines = [
        "# Shutdown Proof Dry-Run Report",
        "",
        f"- task_id: `{TASK_ID}`",
        "- classification: `local_fake_no_order_no_network_dry_run`",
        f"- scenario_count: `{len(rows)}`",
        f"- all_scenarios_passed: `{str(all_passed).lower()}`",
        f"- final_proof_level_counts: `{json.dumps(proof_counts, sort_keys=True)}`",
        "",
        "This dry-run uses fake HBT and fake REST objects only. It does not connect to an exchange, start live trading, place orders, cancel real orders, or call real private/order endpoints.",
        "",
        "## Scenarios",
        "",
    ]
    for row in rows:
        report_lines.append(
            "- `{scenario}` -> `{final_proof_level}` "
            "(exchange_status=`{exchange_reconciliation_status}`, "
            "local_status=`{final_order_status}`, passed=`{passed}`)".format(**row)
        )
    report_path.write_text("\n".join(report_lines) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("local_live_analysis/shutdown_proof_dry_run_0605T004"),
        help="Directory for dry-run artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_dry_run(args.output_dir)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
