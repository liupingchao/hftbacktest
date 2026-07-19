#!/usr/bin/env python3
"""Offline same-window acceptance for the Principal Task 12 tiny-live rerun."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0719T001"
SCHEMA_VERSION = "cross_exchange_principal_task12_same_window_acceptance_v2"
PASSED_RECOMMENDATION = "principal_task12_mechanism_and_evidence_integrity_passed"
BLOCKED_RECOMMENDATION = "principal_task12_same_window_acceptance_blocked"
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "local_live_analysis" / "principal_alignment_task12_repair_0719T001"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "acceptance"
RUNTIME_SOURCE_PROVENANCE_NAME = "runtime_source_provenance.json"
RUNTIME_SOURCE_START_VERIFICATION_NAME = "runtime_source_start_verification.json"
RUNTIME_SOURCE_POSTRUN_VERIFICATION_NAME = "runtime_source_postrun_verification.json"
ALLOWED_ECONOMICS_ONLY_BLOCKERS = {"no_fill_observed"}


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_git_source_snapshot(commit: str) -> tuple[dict[str, str], str]:
    try:
        archive = subprocess.run(
            ["git", "archive", "--format=tar", commit, "examples/hyperliquid"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        expected: dict[str, str] = {}
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            for member in tar.getmembers():
                path = Path(member.name)
                if (
                    not member.isfile()
                    or path.suffix != ".py"
                    or path.name.startswith("test_")
                ):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"git_archive_member_unreadable:{member.name}")
                expected[path.as_posix()] = hashlib.sha256(extracted.read()).hexdigest()
        if not expected:
            return {}, "expected_runtime_source_scope_empty"
        return expected, ""
    except Exception as exc:
        return {}, f"expected_runtime_source_snapshot_failed:{type(exc).__name__}:{exc}"


def runtime_source_digest_map(provenance: dict[str, Any]) -> dict[str, str]:
    rows = provenance.get("files")
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("path", "")): str(row.get("sha256", ""))
        for row in rows
        if isinstance(row, dict) and row.get("path") and row.get("sha256")
    }


def parse_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def check_row(domain: str, check: str, observed: Any, expected: Any, reason: str) -> dict[str, Any]:
    passed = observed == expected
    return {
        "domain": domain,
        "check": check,
        "observed": fmt(observed),
        "expected": fmt(expected),
        "acceptance": "pass" if passed else "fail",
        "reason": reason,
    }


def predicate_row(domain: str, check: str, passed: bool, observed: Any, reason: str) -> dict[str, Any]:
    return {
        "domain": domain,
        "check": check,
        "observed": fmt(observed),
        "expected": "predicate_pass",
        "acceptance": "pass" if passed else "fail",
        "reason": reason,
    }


def command_value(command: list[Any], flag: str) -> str:
    try:
        index = command.index(flag)
    except ValueError:
        return ""
    return str(command[index + 1]) if index + 1 < len(command) else ""


def first_submitted_attempt(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if row.get("side") and row.get("order_status_types") not in {"", "skipped"}:
            return row
    return {}


def btc_position(post_state: dict[str, Any]) -> float | None:
    positions = post_state.get("assetPositions", [])
    if not isinstance(positions, list):
        return None
    total = 0.0
    found = False
    for row in positions:
        position = row.get("position", {}) if isinstance(row, dict) else {}
        if str(position.get("coin", "")) != "BTC":
            continue
        value = parse_float(position.get("szi"))
        if value is None:
            return None
        total += value
        found = True
    return total if found else 0.0


def all_pass(rows: Iterable[dict[str, Any]]) -> bool:
    return all(str(row.get("acceptance", "")) == "pass" for row in rows)


def status_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        status = str(row.get("acceptance", ""))
        result[status] = result.get(status, 0) + 1
    return result


def build_sha256_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.csv":
            rows.append(
                {
                    "artifact": str(path.relative_to(output_dir)),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    write_csv(output_dir / "sha256_manifest.csv", rows, ["artifact", "sha256", "bytes"])


def run_acceptance(
    *,
    input_root: Path,
    output_dir: Path,
    expected_task_id: str,
    expected_source_commit: str,
    expected_max_order_size_btc: float = 0.005,
    expected_max_loss_usdc: float = 1.0,
    expected_max_position_btc: float = 0.01,
    expected_max_submissions: int = 2,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root = input_root / "run"
    window_dir = run_root / "window_01"
    live_dir = window_dir / "window_1" / "pulled_back_awsserver1"

    preflight = read_json(input_root / "preflight" / "orchestrator_preflight.json")
    run_complete = read_json(run_root / "run_complete.json")
    run_status = read_json(run_root / "run_status.json")
    checksum = read_json(run_root / "remote_sha256_verification.json")
    runtime_source = read_json(run_root / RUNTIME_SOURCE_PROVENANCE_NAME)
    runtime_source_start = read_json(run_root / RUNTIME_SOURCE_START_VERIFICATION_NAME)
    runtime_source_postrun = read_json(run_root / RUNTIME_SOURCE_POSTRUN_VERIFICATION_NAME)
    runner_command = read_json(window_dir / "runner_command.json")
    window_status = read_json(window_dir / "window_status.json")
    independent = read_json(window_dir / "independent_remote_open_orders_check.json")
    watcher = read_json(window_dir / "event_driven_watcher_manifest.json")
    estimator = read_json(window_dir / "online_estimator_snapshot.json")
    feedback = read_json(window_dir / "fill_feedback_snapshot.json")
    live_status = read_json(window_dir / "live_status.json")
    config = read_json(live_dir / "approved_config_snapshot.json")
    intent_marker = read_json(live_dir / "run_intent_marker.json")
    fill_manifest = read_json(live_dir / "m2_fill_window_manifest.json")
    executor_manifest = read_json(live_dir / "executor_manifest.json")
    private_response = read_json(live_dir / "private_order_response_audit.json")
    cancel_proof = read_json(live_dir / "cancel_shutdown_proof.json")
    account = read_json(live_dir / "account_inventory_snapshots.json")
    loss = read_json(live_dir / "max_loss_monitor_summary.json")
    fill_pullback = read_json(live_dir / "user_fills_pullback_audit.json")
    attempts = read_csv_rows(live_dir / "quote_attempt_matrix.csv")
    intents = read_csv_rows(live_dir / "order_intent_audit.csv")
    fill_rows = read_csv_rows(live_dir / "live_fill_ledger.csv")
    attribution_rows = read_csv_rows(live_dir / "fill_attribution_evidence.csv")
    role_rows = read_csv_rows(live_dir / "fill_liquidity_role_evidence.csv")
    submitted_attempt = first_submitted_attempt(attempts)
    intent = intents[0] if intents else {}
    command = runner_command.get("command", [])
    if not isinstance(command, list):
        command = []

    source_marker_path = run_root / "source_commit.txt"
    source_marker = source_marker_path.read_text(encoding="utf-8").strip() if source_marker_path.is_file() else ""
    remote_repo_linked = (
        bool(preflight.get("remote_repo"))
        and run_status.get("remote_repo") == preflight.get("remote_repo")
    )
    linked_source_commit = source_marker
    expected_source_digests, expected_source_error = expected_git_source_snapshot(expected_source_commit)
    runtime_source_digests = runtime_source_digest_map(runtime_source)
    expected_source_paths = set(expected_source_digests)
    runtime_source_paths = set(runtime_source_digests)
    identities = {
        "preflight": preflight.get("task_id"),
        "run_complete": run_complete.get("task_id"),
        "run_status": run_status.get("task_id"),
        "window_status": window_status.get("task_id"),
        "independent_open_orders": independent.get("task_id"),
        "watcher": watcher.get("task_id"),
        "approved_config": config.get("task_id"),
        "run_intent": intent_marker.get("task_id"),
        "fill_manifest": fill_manifest.get("task_id"),
        "executor_manifest": executor_manifest.get("task_id"),
    }

    provenance_rows = [
        check_row("provenance", "preflight_status", preflight.get("status"), "pass", "preflight must pass without execution"),
        check_row("provenance", "preflight_only", preflight.get("preflight_only"), True, "preflight must not start watcher"),
        check_row("provenance", "preflight_source_commit", preflight.get("source_commit"), expected_source_commit, "preflight source marker"),
        check_row(
            "provenance",
            "run_remote_repo_matches_preflight",
            remote_repo_linked,
            True,
            "run status points to the exact repository inspected by preflight",
        ),
        check_row(
            "provenance",
            "run_source_commit",
            linked_source_commit,
            expected_source_commit,
            "run-root runtime source marker is mandatory; no path/preflight fallback",
        ),
        check_row("provenance", "runtime_source_status", runtime_source.get("status"), "pass", "runtime source seal passed"),
        check_row("provenance", "runtime_source_task_id", runtime_source.get("task_id"), expected_task_id, "runtime source seal task identity"),
        check_row("provenance", "runtime_source_commit", runtime_source.get("source_commit"), expected_source_commit, "runtime source seal exact commit"),
        check_row("provenance", "runtime_source_marker_origin", runtime_source.get("source_commit_source"), "source_commit.txt", "live archive uses explicit commit marker"),
        check_row("provenance", "runtime_source_sealed_before_watcher", runtime_source.get("sealed_before_watcher_start"), True, "source bytes sealed before child"),
        check_row("provenance", "runtime_source_expected_snapshot_error", expected_source_error, "", "expected source bytes are readable from local Git commit"),
        check_row(
            "provenance",
            "runtime_source_file_set",
            sorted(runtime_source_paths),
            sorted(expected_source_paths),
            "runtime source file set equals expected Git commit scope",
        ),
        check_row("provenance", "runtime_source_start_status", runtime_source_start.get("status"), "pass", "source unchanged immediately before child start"),
        check_row("provenance", "runtime_source_start_phase", runtime_source_start.get("phase"), "pre_watcher_start", "startup verification phase"),
        check_row("provenance", "runtime_source_start_child_not_started", runtime_source_start.get("watcher_process_started"), False, "verification completed before Popen"),
        check_row("provenance", "runtime_source_postrun_status", runtime_source_postrun.get("status"), "pass", "source unchanged after watcher exits"),
        check_row("provenance", "runtime_source_postrun_phase", runtime_source_postrun.get("phase"), "postrun", "terminal source verification phase"),
        check_row("provenance", "run_complete_state", run_complete.get("state"), "complete", "orchestrator terminal state"),
        check_row("provenance", "run_status_state", run_status.get("state"), "complete", "orchestrator status state"),
        check_row("provenance", "checksum_status", checksum.get("status"), "pass", "remote/local terminal manifest verification"),
        check_row("provenance", "checksum_missing_count", checksum.get("missing_count"), 0, "no missing artifact"),
        check_row("provenance", "checksum_mismatch_count", checksum.get("mismatch_count"), 0, "no mismatched artifact"),
    ]
    provenance_rows.extend(
        check_row(
            "runtime_source",
            path,
            runtime_source_digests.get(path, ""),
            expected_digest,
            "sealed runtime bytes equal independently hashed expected Git blob",
        )
        for path, expected_digest in sorted(expected_source_digests.items())
    )
    provenance_rows.extend(
        check_row("identity", name, value, expected_task_id, "all task-owned artifacts share exact identity")
        for name, value in identities.items()
    )
    provenance_rows.extend(
        [
            check_row("identity", "config_window_id", config.get("artifact_window_id"), 1, "exact artifact window"),
            check_row("identity", "intent_window_id", intent_marker.get("artifact_window_id"), 1, "exact artifact window"),
            check_row("identity", "fill_manifest_window_id", fill_manifest.get("artifact_window_id"), 1, "exact artifact window"),
            check_row("identity", "executor_window_id", executor_manifest.get("artifact_window_id"), 1, "exact artifact window"),
        ]
    )

    config_rows = [
        check_row("command", "mode", "--event-driven-live" in command, True, "single-level event-driven live mode"),
        check_row("command", "artifact_task_id", command_value(command, "--artifact-task-id"), expected_task_id, "runner command task identity"),
        check_row("command", "artifact_window_id", command_value(command, "--artifact-window-id"), "1", "runner command window identity"),
        check_row("command", "max_order_size_btc", parse_float(command_value(command, "--max-order-size")), expected_max_order_size_btc, "runner command order cap"),
        check_row("command", "max_loss_usdc", parse_float(command_value(command, "--max-loss-usdc")), expected_max_loss_usdc, "runner command loss cap"),
        check_row("command", "max_position_btc", parse_float(command_value(command, "--max-position-btc")), expected_max_position_btc, "runner command position cap"),
        check_row("command", "max_submissions", command_value(command, "--max-real-order-submissions"), str(expected_max_submissions), "runner command submission cap"),
        check_row("config", "symbol", config.get("symbol"), "BTC", "BTC-only task"),
        check_row("config", "post_only_tif", config.get("time_in_force"), "Alo", "post-only invariant"),
        check_row("config", "reduce_only", config.get("reduce_only"), False, "maker order is not reduce-only"),
        check_row("config", "live_mode", config.get("live_mode"), True, "config records live mode"),
        check_row("config", "max_order_size_btc", parse_float(config.get("max_order_size_btc")), expected_max_order_size_btc, "task-scoped order cap"),
        check_row("config", "max_loss_usdc", parse_float(config.get("max_loss_usdc")), expected_max_loss_usdc, "task-scoped loss cap"),
        check_row("config", "max_position_btc", parse_float(config.get("max_position_btc")), expected_max_position_btc, "task-scoped position cap"),
        check_row("config", "max_submissions", config.get("max_real_order_submissions"), expected_max_submissions, "task-scoped submission cap"),
        check_row("activation", "preflight_activation_off", all(value is False for value in preflight.get("strategy_activation", {}).values()), True, "all preflight activation flags off"),
        check_row("activation", "watcher_dynamic_spread_off", watcher.get("dynamic_spread_activation_enabled"), False, "dynamic spread remains observe-only"),
        check_row("activation", "watcher_quote_behavior_unchanged", watcher.get("actual_quote_behavior_changed"), False, "authoritative fixed quote behavior unchanged"),
        check_row("activation", "estimator_activation_off", estimator.get("activation_enabled"), False, "estimator candidate not activated"),
        check_row("activation", "estimator_quote_behavior_unchanged", estimator.get("actual_quote_behavior_changed"), False, "estimator does not alter quote"),
        check_row("activation", "feedback_dynamic_spread_off", feedback.get("dynamic_spread_activation_enabled"), False, "feedback does not activate dynamic spread"),
        check_row("activation", "feedback_activation_off", feedback.get("fill_feedback_activation_enabled"), False, "fill feedback remains observe-only"),
        check_row("activation", "feedback_quote_behavior_unchanged", feedback.get("actual_quote_behavior_changed"), False, "feedback does not alter quote"),
        check_row("activation", "multi_level_activation_off", live_status.get("risk", {}).get("multi_level_activation_enabled"), False, "single-level task"),
        check_row("status", "writer_health", live_status.get("writer_health", {}).get("status"), "healthy", "status writer healthy"),
        check_row("status", "writer_failure_count", live_status.get("writer_health", {}).get("failure_count"), 0, "no status writer failure"),
        check_row("status", "kill_switch_clear", live_status.get("kill_switch", {}).get("status"), "clear", "kill switch remained clear"),
    ]

    submitted_count = int(watcher.get("live_submissions_count", 0) or 0)
    fill_count = int(watcher.get("fill_count", 0) or 0)
    maker_fill_count = int(watcher.get("maker_fill_count", 0) or 0)
    attempt_keys = [row.get("attempt_key", "") for row in attempts if row.get("attempt_key")]
    decision_rows = [
        check_row("decision", "trigger_found", watcher.get("trigger_found"), True, "same-window public trigger exists"),
        check_row("decision", "event_guard_status", watcher.get("event_driven_guard_status"), "pass", "immediate event guard passed"),
        check_row("decision", "selected_candidate_allowed", watcher.get("selected_candidate", {}).get("fresh_touch_decision", {}).get("allowed"), True, "selected public candidate allowed"),
        predicate_row("decision", "submitted_attempt_present", bool(submitted_attempt), submitted_attempt.get("attempt_key", ""), "at least one live attempt reached order lifecycle"),
        check_row("decision", "intent_side_matches_attempt", intent.get("side"), submitted_attempt.get("side"), "submitted intent/attempt side"),
        check_row("decision", "intent_price_matches_attempt", parse_float(intent.get("limit_px")), parse_float(submitted_attempt.get("limit_px")), "submitted intent/attempt price"),
        check_row("decision", "intent_size_matches_attempt", parse_float(intent.get("size_btc")), parse_float(submitted_attempt.get("size_btc")), "submitted intent/attempt size"),
        check_row("decision", "intent_post_only", intent.get("time_in_force"), "Alo", "submitted intent post-only"),
        predicate_row(
            "decision",
            "submitted_size_within_cap",
            (parse_float(intent.get("size_btc")) or math.inf) <= expected_max_order_size_btc,
            intent.get("size_btc"),
            "actual submitted size stays within cap",
        ),
        predicate_row(
            "identity",
            "attempt_keys_exact",
            bool(attempt_keys) and all(key.startswith(f"{expected_task_id}:window_01:attempt_") for key in attempt_keys),
            ",".join(attempt_keys),
            "attempt identity is task/window scoped",
        ),
    ]

    post_position = btc_position(account.get("post_state", {}))
    estimated_loss = parse_float(loss.get("estimated_loss_usdc"))
    attribution_summary = fill_pullback.get("fill_attribution_summary", {})
    producer_blockers = fill_manifest.get("blocking_reasons", [])
    if not isinstance(producer_blockers, list):
        producer_blockers = ["invalid_blocking_reasons_payload"]
    producer_blockers = [str(reason) for reason in producer_blockers]
    blocker_classification = fill_manifest.get("blocking_reason_classification", {})
    if not isinstance(blocker_classification, dict):
        blocker_classification = {}
    fill_reconciliation = fill_manifest.get("fill_reconciliation", {})
    if not isinstance(fill_reconciliation, dict):
        fill_reconciliation = {}
    permitted_economics_only = {
        reason
        for reason in producer_blockers
        if reason in ALLOWED_ECONOMICS_ONLY_BLOCKERS
        and blocker_classification.get(reason) == "economics_only"
        and fill_reconciliation.get("status") == "no_fill_reconciled"
    }
    unclassified_or_mechanism_blockers = [
        reason
        for reason in producer_blockers
        if reason not in permitted_economics_only
    ]
    lifecycle_rows = [
        check_row("process", "window_state", window_status.get("state"), "complete", "window completed"),
        check_row("process", "child_returncode", window_status.get("child_returncode"), 0, "watcher exited successfully"),
        check_row("process", "child_reaped", window_status.get("child_reaped"), True, "watcher reaped"),
        check_row("process", "no_sigkill", window_status.get("termination_escalated_to_sigkill"), False, "no forced kill"),
        check_row("process", "open_orders_proof_after_exit", window_status.get("open_orders_proof_after_child_exit"), True, "private proof occurs after child exit"),
        predicate_row("lifecycle", "submission_count", 1 <= submitted_count <= expected_max_submissions, submitted_count, "one or two bounded submissions"),
        check_row("lifecycle", "real_order_endpoint_called", fill_manifest.get("real_order_endpoint_called"), True, "real order path observed"),
        check_row("lifecycle", "order_submission_attempted", private_response.get("order_submission_attempted"), True, "private response records submit"),
        check_row("lifecycle", "resting_status_observed", "resting" in fill_manifest.get("order_status_types", []), True, "post-only order reached resting"),
        check_row("lifecycle", "real_cancel_endpoint_called", fill_manifest.get("real_cancel_endpoint_called"), True, "tracked cancellation path observed"),
        check_row("lifecycle", "shutdown_proof_status", fill_manifest.get("shutdown_proof_status"), "pass", "owned-order shutdown proof"),
        check_row("lifecycle", "cancel_proof_status", cancel_proof.get("proof_status"), "pass", "cancel artifact passed"),
        check_row("lifecycle", "final_owned_open_orders", fill_manifest.get("final_open_orders_count"), 0, "window owned orders empty"),
        check_row("lifecycle", "independent_final_open_orders", independent.get("final_open_orders_count"), 0, "independent private proof empty"),
        predicate_row(
            "risk",
            "post_btc_position_within_cap",
            post_position is not None and abs(post_position) <= expected_max_position_btc,
            post_position,
            "post-window BTC position reconciles inside cap",
        ),
        check_row("risk", "max_loss_monitor_status", loss.get("status"), "pass", "max-loss monitor passed"),
        predicate_row(
            "risk",
            "estimated_loss_within_cap",
            estimated_loss is not None and estimated_loss <= expected_max_loss_usdc,
            estimated_loss,
            "estimated loss stays inside task cap",
        ),
        check_row("fills", "watcher_fill_count_matches_manifest", fill_count, int(fill_manifest.get("fill_count", 0) or 0), "fill count agreement"),
        check_row("fills", "ledger_row_count_matches_manifest", len(fill_rows), int(fill_manifest.get("ledger_fill_rows", 0) or 0), "ledger count agreement"),
        check_row("fills", "maker_fill_count_not_over_total", maker_fill_count <= fill_count, True, "maker count cannot exceed total fills"),
        check_row("fills", "unattributed_fill_count", int(attribution_summary.get("unattributed_fill_count", 0) or 0), 0, "no ambiguous/unattributed fill"),
        check_row("producer", "unclassified_or_mechanism_blockers", unclassified_or_mechanism_blockers, [], "acceptance cannot override producer mechanism/evidence blockers"),
    ]
    if fill_count == 0:
        lifecycle_rows.extend(
            [
                check_row("fills", "no_fill_reconciliation_status", fill_reconciliation.get("status"), "no_fill_reconciled", "zero-fill lifecycle is structurally reconciled"),
                check_row("fills", "no_fill_reconciliation_mechanism_status", fill_reconciliation.get("mechanism_status"), "pass", "no-fill mechanism evidence passed"),
                check_row("producer", "zero_fill_blockers", producer_blockers, ["no_fill_observed"], "only the explicit economics-only no-fill blocker remains"),
                check_row("producer", "zero_fill_blocker_classification", blocker_classification.get("no_fill_observed"), "economics_only", "producer classifies no-fill as economics boundary"),
                check_row("producer", "zero_fill_final_recommendation", fill_manifest.get("final_recommendation"), "hyperliquid_tiny_live_m2_fill_window_blocked", "producer remains blocked without fill evidence"),
                check_row("fills", "zero_fill_ledger", len(fill_rows), 0, "zero-fill fact preserved"),
                check_row("fills", "zero_fill_attribution_rows", len(attribution_rows), 0, "no synthetic attribution"),
                check_row("fills", "zero_liquidity_role_rows", len(role_rows), 0, "no synthetic liquidity role"),
            ]
        )
    else:
        lifecycle_rows.extend(
            [
                check_row("producer", "fill_observed_blockers", producer_blockers, [], "filled lifecycle has no producer blocker"),
                check_row("producer", "fill_observed_final_recommendation", fill_manifest.get("final_recommendation"), "hyperliquid_tiny_live_m2_fill_window_ready_for_qa", "producer accepts maker fill lifecycle"),
                check_row("fills", "all_fills_maker", maker_fill_count, fill_count, "all observed fills must be maker"),
                check_row("fills", "liquidity_role_rows_match_fills", len(role_rows), fill_count, "each fill has role evidence"),
            ]
        )

    economics_rows = [
        {
            "domain": "fill_sample",
            "live_evidence": f"fill_count={fill_count}",
            "boundary": "single_window_single_digit_only",
            "acceptance": "pass",
            "reason": "mechanism evidence does not establish stable fill rate",
        },
        {
            "domain": "fee_rebate",
            "live_evidence": f"liquidity_role_rows={len(role_rows)}",
            "boundary": "supported_per_fill_only" if fill_count else "unsupported_no_fill",
            "acceptance": "pass",
            "reason": "do not infer fee/rebate calibration beyond observed fills",
        },
        {
            "domain": "pnl",
            "live_evidence": f"estimated_loss_usdc={estimated_loss}",
            "boundary": "risk_observation_not_stable_economics",
            "acceptance": "pass",
            "reason": "loss monitor pass is not profitability proof",
        },
        {
            "domain": "queue_priority",
            "live_evidence": "public flow and resting lifecycle only",
            "boundary": "unsupported",
            "acceptance": "pass",
            "reason": "no exact queue-position claim",
        },
        {
            "domain": "maker_viability",
            "live_evidence": f"submissions={submitted_count};fills={fill_count}",
            "boundary": "unsupported",
            "acceptance": "pass",
            "reason": "one tiny-live window cannot establish maker viability",
        },
        {
            "domain": "multi_level",
            "live_evidence": "single_level_only",
            "boundary": "not_activated",
            "acceptance": "pass",
            "reason": "later formal task must reconsider the prerequisite",
        },
    ]
    optimism_rows = [
        {"check": "no_synthetic_fill", "evidence": f"ledger_rows={len(fill_rows)}", "acceptance": "pass"},
        {"check": "no_zero_fill_probability_claim", "evidence": "one window is not a calibrated denominator", "acceptance": "pass"},
        {"check": "no_profitability_claim", "evidence": "risk pass only", "acceptance": "pass"},
        {"check": "no_queue_priority_claim", "evidence": "public proxy is not exact queue", "acceptance": "pass"},
        {"check": "no_multi_level_activation", "evidence": "single-level authoritative behavior", "acceptance": "pass"},
        {"check": "no_promotion_claim", "evidence": "mechanism/evidence gate only", "acceptance": "pass"},
    ]

    write_csv(
        output_dir / "provenance_identity_comparison.csv",
        provenance_rows,
        ["domain", "check", "observed", "expected", "acceptance", "reason"],
    )
    write_csv(
        output_dir / "config_control_comparison.csv",
        config_rows,
        ["domain", "check", "observed", "expected", "acceptance", "reason"],
    )
    write_csv(
        output_dir / "decision_replay_comparison.csv",
        decision_rows,
        ["domain", "check", "observed", "expected", "acceptance", "reason"],
    )
    write_csv(
        output_dir / "lifecycle_evidence_comparison.csv",
        lifecycle_rows,
        ["domain", "check", "observed", "expected", "acceptance", "reason"],
    )
    write_csv(
        output_dir / "economics_boundary_matrix.csv",
        economics_rows,
        ["domain", "live_evidence", "boundary", "acceptance", "reason"],
    )
    write_csv(output_dir / "optimism_check_matrix.csv", optimism_rows, ["check", "evidence", "acceptance"])

    mechanism_rows = provenance_rows + config_rows + decision_rows + lifecycle_rows
    mechanism_pass = all_pass(mechanism_rows)
    boundary_pass = all_pass(economics_rows) and all_pass(optimism_rows)
    final_pass = mechanism_pass and boundary_pass
    manifest = {
        "task_id": expected_task_id,
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "git_commit": git_commit(),
        "input_root": str(input_root),
        "expected_source_commit": expected_source_commit,
        "final_recommendation": PASSED_RECOMMENDATION if final_pass else BLOCKED_RECOMMENDATION,
        "mechanism_and_evidence_integrity_acceptance": "pass" if mechanism_pass else "fail",
        "economics_boundary_acceptance": "pass" if boundary_pass else "fail",
        "provenance_identity_counts": status_counts(provenance_rows),
        "config_control_counts": status_counts(config_rows),
        "decision_replay_counts": status_counts(decision_rows),
        "lifecycle_evidence_counts": status_counts(lifecycle_rows),
        "economics_boundary_counts": status_counts(economics_rows),
        "optimism_check_counts": status_counts(optimism_rows),
        "live_summary": {
            "submissions": submitted_count,
            "fill_count": fill_count,
            "maker_fill_count": maker_fill_count,
            "final_owned_open_orders": fill_manifest.get("final_open_orders_count"),
            "independent_final_open_orders": independent.get("final_open_orders_count"),
            "post_btc_position": post_position,
            "estimated_loss_usdc": estimated_loss,
        },
        "supported_claims": [
            "task-scoped envelope enforcement",
            "task/window/attempt identity integrity",
            "single-level post-only submit/resting/cancel lifecycle",
            "terminal open-orders/account/checksum reconciliation",
            "same-window config/decision/control reproduction",
        ] if final_pass else [],
        "unsupported_claims": [
            "stable pnl",
            "fill-rate calibration",
            "fee/rebate calibration beyond observed fills",
            "queue priority",
            "maker viability",
            "multi-level activation",
            "promotion",
            "final mvp pass",
        ],
        "multi_level_activation_unlocked": False,
        "boundary": {
            "offline_only": True,
            "network_called": False,
            "remote_called": False,
            "credentials_read": False,
            "private_endpoint_called": False,
            "order_endpoint_called": False,
            "cancel_endpoint_called": False,
            "new_live_window_started": False,
        },
    }
    write_json(output_dir / "same_window_acceptance_manifest.json", manifest)
    report = [
        "# 0718T024 Same-Window Acceptance",
        "",
        f"Final recommendation: `{manifest['final_recommendation']}`",
        "",
        f"- Mechanism/evidence integrity: `{manifest['mechanism_and_evidence_integrity_acceptance']}`",
        f"- Economics boundary: `{manifest['economics_boundary_acceptance']}`",
        f"- Live summary: `{manifest['live_summary']}`",
        "",
        "This offline acceptance keeps live order/fill facts authoritative. It does not infer stable economics, queue priority, maker viability, multi-level activation, promotion, or final MVP pass.",
        "",
    ]
    (output_dir / "validation_report.md").write_text("\n".join(report), encoding="utf-8")
    build_sha256_manifest(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-task-id", default=TASK_ID)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-max-order-size-btc", type=float, default=0.005)
    parser.add_argument("--expected-max-loss-usdc", type=float, default=1.0)
    parser.add_argument("--expected-max-position-btc", type=float, default=0.01)
    parser.add_argument("--expected-max-submissions", type=int, default=2)
    args = parser.parse_args()
    manifest = run_acceptance(
        input_root=args.input_root,
        output_dir=args.output_dir,
        expected_task_id=args.expected_task_id,
        expected_source_commit=args.expected_source_commit,
        expected_max_order_size_btc=args.expected_max_order_size_btc,
        expected_max_loss_usdc=args.expected_max_loss_usdc,
        expected_max_position_btc=args.expected_max_position_btc,
        expected_max_submissions=args.expected_max_submissions,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["final_recommendation"] == PASSED_RECOMMENDATION else 2


if __name__ == "__main__":
    raise SystemExit(main())
