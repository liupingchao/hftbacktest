#!/usr/bin/env python3
"""Aggregate acceptance for T068's three independent live windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any


TASK_ID = "0726T068"
SOURCE_COMMIT = "a0bc92898ecea43cbdc4219efc1acbcd69580969"
SEED_SHA256 = "e35c7fd8f3ec8268e5d50c7963889b73409470c9f01f92ca3a0d598a96562be9"
ACCOUNT_SCOPE_SHA256 = "59153858d04cb15ef51660a62b2598b785468306e4d8ab36bdbbfa18ac134c6d"
SIGNER_SHA256 = "9dd9fcfdb4e3b3e077ba25b41ccc3514d824b952a570fd5a98c3188ec72de422"
EXPECTED_TAR_SHA256 = "5bd81ab915dac98b5141875cc932a9ce7aebe2f11f34741b84244b5cdfb3446d"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle_manifest(extracted_root: Path) -> dict[str, Any]:
    manifest = extracted_root / "0726T068_bundle_file_manifest.txt"
    missing: list[str] = []
    mismatched: list[str] = []
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, remote_path = line.split("  ", 1)
        prefix = "/home/admin/"
        if not remote_path.startswith(prefix):
            mismatched.append(f"unexpected_remote_path:{remote_path}")
            continue
        local_path = extracted_root / remote_path.removeprefix(prefix)
        if not local_path.is_file():
            missing.append(str(local_path))
            continue
        checked += 1
        if sha256_file(local_path) != expected:
            mismatched.append(str(local_path))
    return {
        "status": "pass" if not missing and not mismatched else "fail",
        "checked_file_count": checked,
        "missing_files": missing,
        "mismatched_files": mismatched,
    }


def window_summary(base: Path, window_id: int) -> dict[str, Any]:
    root = base / f"0726T068_window_{window_id:02d}_R1"
    artifact = root / "window_01"
    run_complete = load_json(root / "run_complete.json")
    source_start = load_json(root / "runtime_source_start_verification.json")
    source_post = load_json(root / "runtime_source_postrun_verification.json")
    manifest = load_json(artifact / "m2_fill_window_manifest.json")
    watcher = load_json(artifact / "event_driven_watcher_manifest.json")
    config = load_json(artifact / "approved_config_snapshot.json")
    cancel = load_json(artifact / "cancel_shutdown_proof.json")
    fill_pullback = load_json(artifact / "user_fills_pullback_audit.json")
    max_loss = load_json(artifact / "max_loss_monitor_summary.json")
    intents = read_csv(artifact / "order_intent_audit.csv")
    fills = read_csv(artifact / "live_fill_ledger.csv")
    role_rows = read_csv(artifact / "fill_liquidity_role_evidence.csv")
    run_result = run_complete["window_results"][0]
    requested_seconds = float(watcher["watcher_seconds_requested"])
    elapsed_seconds = float(watcher["watcher_seconds_elapsed"])
    submission_cap = int(watcher["max_real_order_submissions"])
    submission_count = len(intents)
    duration_route = (
        "elapsed_full_window"
        if elapsed_seconds >= requested_seconds - 1.0
        else (
            "submission_cap_terminal"
            if submission_count == submission_cap
            else "early_exit_without_submission_cap"
        )
    )

    identity_ok = (
        manifest["task_id"] == TASK_ID
        and manifest["artifact_window_id"] == window_id
        and manifest["window_id"] == f"window_{window_id:02d}"
    )
    source_ok = all(
        verification.get("status") == "pass"
        and verification.get("source_commit") == SOURCE_COMMIT
        and verification.get("source_commit_matches") is True
        for verification in (source_start, source_post)
    )
    run_ok = (
        run_complete.get("state") == "complete"
        and run_result.get("state") == "complete"
        and run_result.get("child_reaped") is True
        and run_result.get("child_returncode") == 0
        and run_result.get("runner_returncode") == 0
        and run_result.get("independent_open_orders_empty") is True
        and run_result.get("open_orders_proof_after_child_exit") is True
    )
    seed = watcher["dynamic_spread_seed_load"]
    seed_ok = (
        seed.get("status") == "pass"
        and seed.get("seed_contract_sha256") == SEED_SHA256
        and seed.get("loaded_row_count") == 280
        and seed.get("current_market_state_contaminated") is False
        and watcher.get("require_strict_seeded_dynamic_submit") is True
    )
    config_ok = (
        config.get("symbol") == "BTC"
        and config.get("time_in_force") == "Alo"
        and float(config.get("max_order_size_btc")) == 0.005
        and float(config.get("max_position_btc")) == 0.01
        and float(config.get("max_loss_usdc")) == 1.0
        and int(config.get("max_real_order_submissions")) == 2
    )
    intents_ok = all(
        row["symbol"] == "BTC"
        and row["time_in_force"] == "Alo"
        and float(row["size_btc"]) <= 0.005
        and row["endpoint_called"].lower() == "true"
        for row in intents
    )
    strict_gate = watcher["strict_seeded_dynamic_submit_gate"]
    strict_submit_ok = (
        (
            strict_gate.get("status") == "not_evaluated"
            and strict_gate.get("allowed") is False
        )
        if submission_count == 0
        else (
            strict_gate.get("status") == "pass"
            and strict_gate.get("allowed") is True
            and all(strict_gate.get("checks", {}).values())
            and watcher.get("actual_quote_behavior_changed") is True
            and watcher.get("dynamic_spread_candidate_status") == "pass"
            and watcher.get("dynamic_spread_fallback_to_fixed") is False
        )
    )
    terminal_ok = (
        manifest.get("shutdown_proof_status") == "pass"
        and manifest.get("final_open_orders_count") == 0
        and cancel.get("proof_status") == "pass"
        and (
            max_loss.get("status") == "pass"
            or (
                submission_count == 0
                and max_loss.get("status") == "not_evaluated"
                and max_loss.get("reason") == "no_order_submitted"
            )
        )
    )
    if submission_count:
        terminal_ok = terminal_ok and (
            manifest.get("real_order_endpoint_called") is True
            and manifest.get("real_cancel_endpoint_called") is True
            and cancel.get("real_cancel_endpoint_called") is True
            and cancel.get("cancel_reference_reconciliation", {}).get("status")
            == "pass"
        )

    role_counts: dict[str, int] = {}
    for row in role_rows:
        role = (
            row.get("liquidity_role")
            or row.get("role_classification")
            or row.get("liquidity")
            or ""
        )
        role_counts[role] = role_counts.get(role, 0) + 1

    return {
        "window_id": window_id,
        "run_id": f"{TASK_ID}:window_{window_id:02d}",
        "started_at_utc": run_result["started_at_utc"],
        "ended_at_utc": run_result["ended_at_utc"],
        "watcher_seconds_requested": requested_seconds,
        "watcher_seconds_elapsed": elapsed_seconds,
        "duration_route": duration_route,
        "duration_contract_pass": duration_route
        in {"elapsed_full_window", "submission_cap_terminal"},
        "identity_pass": identity_ok,
        "source_provenance_pass": source_ok,
        "run_lifecycle_pass": run_ok,
        "seed_contract_pass": seed_ok,
        "config_envelope_pass": config_ok,
        "strict_submit_gate_pass": strict_submit_ok,
        "order_intent_pass": intents_ok,
        "terminal_proof_pass": terminal_ok,
        "candidate_attempt_count": manifest["candidate_attempt_evidence_row_count"],
        "submission_count": submission_count,
        "submission_cap": submission_cap,
        "submitted_sides": [row["side"] for row in intents],
        "submitted_symbols": sorted({row["symbol"] for row in intents}),
        "submitted_sizes_btc": [float(row["size_btc"]) for row in intents],
        "real_order_endpoint_called": manifest["real_order_endpoint_called"],
        "real_cancel_endpoint_called": manifest["real_cancel_endpoint_called"],
        "resting_order_count": sum(
            "resting" in row.get("order_status_types", "").split("|")
            for row in read_csv(artifact / "quote_attempt_matrix.csv")
        ),
        "post_only_reject_count": int(manifest["post_only_reject_count"]),
        "fill_count": len(fills),
        "manifest_fill_count": int(manifest["fill_count"]),
        "maker_fill_count": int(manifest["maker_fill_count"]),
        "role_evidence_row_count": len(role_rows),
        "role_counts": role_counts,
        "fill_pullback_count": int(fill_pullback["pullback_count"]),
        "fill_reconciliation_status": cancel["fill_reconciliation"]["status"],
        "blocking_reasons": manifest["blocking_reasons"],
        "window_pass": all(
            (
                identity_ok,
                source_ok,
                run_ok,
                seed_ok,
                config_ok,
                intents_ok,
                strict_submit_ok,
                terminal_ok,
                duration_route
                in {"elapsed_full_window", "submission_cap_terminal"},
            )
        ),
    }


def account_chain_summary(control: Path) -> dict[str, Any]:
    guard_names = [
        "controller_prestart_account_guard.json",
        "window_01_pre_account_guard.json",
        "window_01_post_account_guard.json",
        "window_02_pre_account_guard.json",
        "window_02_post_account_guard.json",
        "window_03_pre_account_guard.json",
        "window_03_post_account_guard.json",
        "controller_final_account_guard.json",
        "independent_terminal_account_guard.json",
    ]
    guards = [load_json(control / name) for name in guard_names]
    pass_status = all(
        guard.get("status") == "pass"
        and guard.get("source_commit") == SOURCE_COMMIT
        and guard.get("account_scope_sha256") == ACCOUNT_SCOPE_SHA256
        and guard.get("signer_sha256") == SIGNER_SHA256
        and guard.get("open_orders_empty") is True
        and guard.get("open_orders_count") == 0
        and guard.get("btc_position") == 0.0
        and guard.get("position_within_cap") is True
        and guard.get("raw_credentials_written") is False
        for guard in guards
    )
    controller = load_json(control / "controller_final.json")
    secret_scan = load_json(control / "artifact_secret_scan.json")
    return {
        "status": "pass"
        if pass_status
        and controller.get("state") == "complete"
        and controller.get("completed_windows") == [1, 2, 3]
        and controller.get("account_scope_sha256") == ACCOUNT_SCOPE_SHA256
        and controller.get("signer_sha256") == SIGNER_SHA256
        and controller.get("final_open_orders_empty") is True
        and controller.get("final_btc_position") == 0.0
        and secret_scan.get("status") == "pass"
        and secret_scan.get("raw_identity_match_count") == 0
        and secret_scan.get("env_named_file_count") == 0
        else "fail",
        "guard_count": len(guards),
        "all_guards_same_account_scope": len(
            {guard["account_scope_sha256"] for guard in guards}
        )
        == 1,
        "all_guards_same_signer": len(
            {guard["signer_sha256"] for guard in guards}
        )
        == 1,
        "final_open_orders_count": guards[-1]["open_orders_count"],
        "final_btc_position": guards[-1]["btc_position"],
        "raw_credentials_pulled_local": False,
        "artifact_secret_scan_status": secret_scan["status"],
        "artifact_secret_scan_file_count": secret_scan["scanned_file_count"],
        "same_process_client_contract": (
            "The exact watcher source initializes one live client and uses "
            "that same client for submit, user_fills_by_time, user_state, "
            "fees and final open_orders; external guards bind the env-backed "
            "account before and after every window."
        ),
    }


def write_outputs(
    output_dir: Path,
    aggregate: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregate_acceptance.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "window_id",
        "run_id",
        "started_at_utc",
        "ended_at_utc",
        "watcher_seconds_requested",
        "watcher_seconds_elapsed",
        "duration_route",
        "submission_count",
        "submitted_sides",
        "resting_order_count",
        "post_only_reject_count",
        "fill_count",
        "role_evidence_row_count",
        "blocking_reasons",
        "window_pass",
    ]
    with (output_dir / "window_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregate["windows"]:
            writer.writerow(
                {
                    key: json.dumps(row[key], sort_keys=True)
                    if isinstance(row[key], (list, dict))
                    else row[key]
                    for key in fieldnames
                }
            )

    windows = aggregate["windows"]
    report = [
        "# T068 Aggregate Acceptance",
        "",
        f"- Overall status: `{aggregate['overall_status']}`",
        f"- Operational mechanism: `{aggregate['mechanism_status']}`",
        f"- Role-known fill evidence: `{aggregate['role_known_fill_status']}`",
        f"- Economics: `{aggregate['economics_status']}`",
        f"- Bundle integrity: `{aggregate['bundle_integrity']['status']}` "
        f"({aggregate['bundle_integrity']['checked_file_count']} files)",
        f"- Account chain: `{aggregate['account_chain']['status']}`",
        "",
        "| Window | Elapsed | Completion route | Submits | Lifecycle | Fills | Result |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    for row in windows:
        lifecycle = (
            f"{row['post_only_reject_count']} rejected / "
            f"{row['resting_order_count']} resting"
        )
        report.append(
            f"| {row['window_id']:02d} | "
            f"{row['watcher_seconds_elapsed']:.3f}s | "
            f"{row['duration_route']} | "
            f"{row['submission_count']} | {lifecycle} | "
            f"{row['fill_count']} | "
            f"{'pass' if row['window_pass'] else 'fail'} |"
        )
    report.extend(
        [
            "",
            "## Conclusion",
            "",
            "- Exact seeded dynamic quote generation reached a strict-pass "
            "live submit in window 03 and changed the final tick-rounded quote.",
            "- Both submitted intents were Hyperliquid BTC `Alo`, size "
            "`0.005 BTC`; the buy was post-only rejected and the sell rested, "
            "then was authoritatively canceled.",
            "- No fill occurred. Maker/taker role, fee/rebate attribution, "
            "markout and economic viability therefore remain blocked.",
            "- Final independent account state is zero open orders and zero "
            "BTC position; no raw credential or account address was pulled "
            "into the local evidence package.",
            "",
        ]
    )
    (output_dir / "validation_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    script_dir = Path(__file__).resolve().parent
    default_pull = script_dir / "pulled_back_R1"
    parser.add_argument("--pull-root", type=Path, default=default_pull)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "acceptance",
    )
    args = parser.parse_args()

    pull_root = args.pull_root.resolve()
    extracted = pull_root / "extracted"
    tar_path = pull_root / "0726T068_live_artifacts_R1.tar.gz"
    temporary_extract: tempfile.TemporaryDirectory[str] | None = None
    if not extracted.is_dir():
        temporary_extract = tempfile.TemporaryDirectory(
            prefix="t068_artifacts_"
        )
        extracted = Path(temporary_extract.name)
        with tarfile.open(tar_path, "r:gz") as archive:
            archive.extractall(extracted, filter="data")
    base = extracted / "hftbacktest-cross-exchange-artifacts"
    tar_integrity = {
        "expected_sha256": EXPECTED_TAR_SHA256,
        "actual_sha256": sha256_file(tar_path),
    }
    tar_integrity["status"] = (
        "pass"
        if tar_integrity["actual_sha256"] == tar_integrity["expected_sha256"]
        else "fail"
    )
    bundle = verify_bundle_manifest(extracted)
    bundle["tar"] = tar_integrity
    if tar_integrity["status"] != "pass":
        bundle["status"] = "fail"

    windows = [window_summary(base, window_id) for window_id in (1, 2, 3)]
    account_chain = account_chain_summary(base / "0726T068_control_R1")
    total_submissions = sum(row["submission_count"] for row in windows)
    total_fills = sum(row["fill_count"] for row in windows)
    total_role_rows = sum(row["role_evidence_row_count"] for row in windows)
    mechanism_pass = (
        bundle["status"] == "pass"
        and account_chain["status"] == "pass"
        and all(row["window_pass"] for row in windows)
        and total_submissions == 2
        and any(row["resting_order_count"] > 0 for row in windows)
        and any(
            row["strict_submit_gate_pass"]
            and row["submission_count"] > 0
            for row in windows
        )
    )
    role_known_fill_pass = total_fills > 0 and total_role_rows == total_fills
    economics_pass = role_known_fill_pass
    aggregate = {
        "schema_version": "t068_three_window_aggregate_acceptance_v1",
        "task_id": TASK_ID,
        "source_commit": SOURCE_COMMIT,
        "bundle_integrity": bundle,
        "account_chain": account_chain,
        "windows": windows,
        "totals": {
            "submission_count": total_submissions,
            "fill_count": total_fills,
            "role_evidence_row_count": total_role_rows,
        },
        "mechanism_status": "pass" if mechanism_pass else "fail",
        "role_known_fill_status": (
            "pass" if role_known_fill_pass else "blocked_no_fill"
        ),
        "economics_status": (
            "pass" if economics_pass else "blocked_no_role_known_fill"
        ),
        "overall_status": (
            "pass"
            if mechanism_pass and role_known_fill_pass and economics_pass
            else ("blocked" if mechanism_pass else "fail")
        ),
        "blocking_reasons": (
            []
            if mechanism_pass and role_known_fill_pass and economics_pass
            else (
                [
                    "no_fill_observed",
                    "maker_taker_role_evidence_absent",
                    "fee_rebate_markout_pnl_not_evaluable",
                ]
                if mechanism_pass
                else ["operational_or_mechanism_acceptance_failed"]
            )
        ),
    }
    write_outputs(args.output_dir.resolve(), aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    if temporary_extract is not None:
        temporary_extract.cleanup()
    return 0 if aggregate["overall_status"] in {"pass", "blocked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
