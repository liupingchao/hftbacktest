from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

try:
    import research_package_trust as trust
    import skhynix_stage_h0a as h0a
    import skhynix_stage_h0a_support as support
except ModuleNotFoundError:  # pragma: no cover
    from examples.hyperliquid import research_package_trust as trust
    from examples.hyperliquid import skhynix_stage_h0a as h0a
    from examples.hyperliquid import skhynix_stage_h0a_support as support


def _session_row(
    session_id: str,
    horizon_ms: int,
    *,
    passes: bool,
) -> dict[str, str]:
    spec = next(
        item for item in support.SESSION_SPECS if item.session_id == session_id
    )
    return {
        "schema_version": support.SCHEMA_VERSION,
        "session_id": session_id,
        "evidence_label": spec.evidence_label,
        "formal_eligible": support.bool_text(spec.formal_eligible),
        "horizon_ms": str(horizon_ms),
        "primary_selection_eligible": "true",
        "segment_count": str(spec.segment_count),
        "connection_epoch_count": str(spec.segment_count),
        "nominal_calendar_grid_count": "100",
        "quality_eligible_grid_count": "100",
        "quality_eligible_calendar_exposure_fraction": "1.000000000000",
        "fully_identified_binary_count": "100",
        "fully_identified_binary_endpoint_fraction": "1.000000000000",
        "interval_likelihood_eligible_count": "100",
        "interval_likelihood_eligible_fraction": "1.000000000000",
        "complete_60s_block_count": "20" if passes else "19",
        "quality_exposure_gate_pass": "true",
        "binary_identification_gate_pass": "true",
        "interval_likelihood_gate_pass": "true",
        "complete_block_gate_pass": support.bool_text(passes),
        "session_support_gate_pass": support.bool_text(passes),
    }


def test_selector_chooses_first_cross_session_pass() -> None:
    rows = []
    for horizon in support.PRIMARY_HORIZONS_MS:
        for spec in support.SESSION_SPECS:
            formal_pass = horizon >= 100
            passes = formal_pass if spec.formal_eligible else True
            rows.append(_session_row(spec.session_id, horizon, passes=passes))
    trace, status, selected = h0a._selection_trace(rows)
    assert status == "selected"
    assert selected == 100
    assert len(trace) == 12
    assert not any(
        row["selected_at_this_horizon"] and row["horizon_ms"] == 50
        for row in trace
    )


def test_primary_tuple_c_binding_is_normalized_out_of_r(
    tmp_path: Path,
) -> None:
    for relative in h0a.R_FILES:
        path = tmp_path / relative
        if relative == "primary_tuple_freeze.json":
            path.write_text(
                json.dumps({"code_contract_identity": "a" * 64}) + "\n",
                encoding="ascii",
            )
        else:
            path.write_text(relative + "\n", encoding="ascii")
    before = h0a.research_identity(tmp_path)
    (tmp_path / "primary_tuple_freeze.json").write_text(
        json.dumps({"code_contract_identity": "b" * 64}) + "\n",
        encoding="ascii",
    )
    assert h0a.research_identity(tmp_path) == before


def test_manifest_identity_fields_are_normalized_out_of_e(
    tmp_path: Path,
) -> None:
    for relative in h0a.E_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "h0a_manifest.json":
            payload = {
                "research_data_identity": "a" * 64,
                "code_contract_identity": "b" * 64,
                "evidence_identity": "c" * 64,
                "composite_identity": "d" * 64,
            }
            path.write_text(json.dumps(payload) + "\n", encoding="ascii")
        else:
            path.write_text(relative + "\n", encoding="ascii")
    before = h0a.evidence_inventory(tmp_path)
    manifest = json.loads(
        (tmp_path / "h0a_manifest.json").read_text(encoding="ascii")
    )
    manifest["composite_identity"] = "e" * 64
    (tmp_path / "h0a_manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="ascii"
    )
    assert h0a.evidence_inventory(tmp_path) == before


def test_atomic_publication_refuses_existing_final(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    final = tmp_path / "final"
    staging.mkdir()
    final.mkdir()
    with pytest.raises(trust.TrustKernelError) as error:
        trust.publish_atomically(staging, final)
    assert error.value.code == "PUBLICATION_FINAL_EXISTS"


def test_metadata_quiescence_waits_for_stable_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = iter(
        [
            [{"path": ".", "ctime_ns": 1}],
            [{"path": ".", "ctime_ns": 2}],
            [{"path": ".", "ctime_ns": 2}],
        ]
    )
    monkeypatch.setattr(
        h0a.trust,
        "metadata_snapshot",
        lambda _root: next(samples),
    )
    monkeypatch.setattr(h0a.time, "sleep", lambda _seconds: None)
    h0a._wait_for_metadata_quiescence(
        tmp_path,
        timeout_seconds=1.0,
        poll_seconds=0.0,
    )


def test_selector_rejects_extra_raw_path(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    for name in h0a.BASE_PROJECTION_FILES:
        (sealed / name).write_text("", encoding="ascii")
    for name in (
        "sealed_projection.json",
        "selector_context.json",
        "task.md",
        "surface_matrix.json",
        "raw.csv",
    ):
        (sealed / name).write_text("{}\n", encoding="ascii")
    with pytest.raises(support.H0AError) as error:
        h0a.run_selector(sealed, tmp_path / "out")
    assert error.value.code == "H0A_SELECTOR_INPUT_UNIVERSE_MISMATCH"


def test_write_csv_uses_lowercase_booleans(tmp_path: Path) -> None:
    path = tmp_path / "value.csv"
    support.write_csv(path, [{"flag": True}], ("flag",))
    with path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == [["flag"], ["true"]]
