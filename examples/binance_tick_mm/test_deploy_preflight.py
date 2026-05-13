from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parent / "deploy" / "preflight_live_run.py"
SPEC = importlib.util.spec_from_file_location("preflight_live_run", MODULE_PATH)
assert SPEC is not None
preflight_live_run = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(preflight_live_run)


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = ROOT / "examples" / "binance_tick_mm"


def test_validate_audit_row_keys_accepts_current_strategy_schema() -> None:
    strategy_core = preflight_live_run.load_strategy_core(EXAMPLE_DIR)
    decision_row = preflight_live_run.build_sample_decision_row(strategy_core, "BTCUSDT")
    result = preflight_live_run.validate_audit_row_keys(
        list(strategy_core.AUDIT_FIELDS),
        list(decision_row.keys()),
        label="decision",
    )

    assert result["field_count"] == len(strategy_core.AUDIT_FIELDS)
    assert result["row_key_count"] == len(strategy_core.AUDIT_FIELDS)
    assert result["missing"] == []
    assert result["extra"] == []


def test_validate_audit_row_keys_detects_missing_and_extra_fields() -> None:
    with pytest.raises(preflight_live_run.PreflightError, match="missing=.*b.*extra=.*c"):
        preflight_live_run.validate_audit_row_keys(["a", "b"], ["a", "c"], label="unit")


def test_validate_audit_row_keys_detects_duplicate_fields() -> None:
    with pytest.raises(preflight_live_run.PreflightError, match="duplicates=.*a"):
        preflight_live_run.validate_audit_row_keys(["a", "a"], ["a"], label="unit")


def test_run_preflight_writes_manifest_and_start_marker(tmp_path: Path) -> None:
    manifest_out = tmp_path / "run" / "deployment_manifest.json"
    start_marker = tmp_path / "run" / "start_marker.json"
    stop_marker = tmp_path / "run" / "stop_marker.json"

    manifest = preflight_live_run.run_preflight(
        project_root=ROOT,
        config=EXAMPLE_DIR / "config.example.toml",
        connector_config=EXAMPLE_DIR / "deploy" / "binancefutures.toml",
        symbol="BTCUSDT",
        data_dir=tmp_path / "run" / "data",
        run_dir=tmp_path / "run",
        manifest_out=manifest_out,
        start_marker_out=start_marker,
        stop_marker_out=stop_marker,
    )

    saved = json.loads(manifest_out.read_text(encoding="utf-8"))
    marker = json.loads(start_marker.read_text(encoding="utf-8"))
    assert saved["decision"] == "preflight_passed"
    assert saved["git"]["short_commit"]
    assert saved["hashes"]["config"]
    assert saved["hashes"]["audit_schema"]
    assert saved["hashes"]["strategy_core"]
    assert saved["compatibility"]["passed"] is True
    assert saved["compatibility"]["decision_row"]["missing"] == []
    assert saved["compatibility"]["lifecycle_row"]["extra"] == []
    assert marker["event"] == "preflight_passed_before_tmux_start"
    assert marker["manifest"] == str(manifest_out.resolve())
    assert manifest["inputs"]["manifest_out"] == str(manifest_out.resolve())


def test_write_stop_marker_uses_existing_manifest(tmp_path: Path) -> None:
    manifest_out = tmp_path / "deployment_manifest.json"
    stop_marker = tmp_path / "stop_marker.json"
    manifest_out.write_text(
        json.dumps(
            {
                "host": "unit-host",
                "git": {"short_commit": "abc123"},
                "inputs": {"symbol": "BTCUSDT"},
            }
        ),
        encoding="utf-8",
    )

    marker = preflight_live_run.write_stop_marker(stop_marker, manifest_out, 7)

    saved = json.loads(stop_marker.read_text(encoding="utf-8"))
    assert marker["exit_code"] == 7
    assert saved["event"] == "live_bot_process_exited"
    assert saved["git"]["short_commit"] == "abc123"
    assert saved["symbol"] == "BTCUSDT"
