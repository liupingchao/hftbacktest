from __future__ import annotations

from pathlib import Path

import pytest

from examples.tutorial_reproduction import notebook_support


def test_experiment_mapping_covers_nine_notebooks() -> None:
    assert [slug for slug, _, _ in notebook_support.EXPERIMENTS] == [
        "getting_started",
        "working_with_market_depth_and_trades",
        "data_preparation",
        "fusing_depth_data",
        "order_latency_data",
        "impact_of_order_latency",
        "accelerated_backtesting",
        "level_3_backtesting",
        "integrating_custom_data",
    ]
    assert len({notebook for _, notebook, _ in notebook_support.EXPERIMENTS}) == 9


def test_notebook_context_prefers_configured_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HFTBACKTEST_TARDIS_ROOT", str(tmp_path))
    monkeypatch.setenv("HFTBACKTEST_TARDIS_DATE", "2025-05-01")
    monkeypatch.setenv("HFTBACKTEST_NOTEBOOK_SECONDS", "60")
    monkeypatch.setenv("HFTBACKTEST_NOTEBOOK_OUTPUT", str(tmp_path / "output"))

    context = notebook_support.notebook_context()

    assert context.tardis_root == str(tmp_path.resolve())
    assert context.date == "2025-05-01"
    assert context.duration_seconds == 60
    assert context.output_root == str((tmp_path / "output").resolve())


def test_notebook_context_rejects_too_short_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HFTBACKTEST_TARDIS_ROOT", str(tmp_path))
    monkeypatch.setenv("HFTBACKTEST_NOTEBOOK_SECONDS", "29")

    with pytest.raises(ValueError, match="at least 30"):
        notebook_support.notebook_context()
