from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from examples.tutorial_reproduction import notebook_support

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def test_generated_notebooks_have_a_safe_run_all_boundary() -> None:
    for slug, notebook_name, _ in notebook_support.EXPERIMENTS:
        path = PROJECT_ROOT / "examples" / notebook_name
        notebook = json.loads(path.read_text())
        cells = notebook["cells"]
        metadata = notebook["metadata"]["tutorial_reproduction"]
        assert metadata["generated"] is True
        assert metadata["slug"] == slug

        code_indexes = [
            index for index, cell in enumerate(cells) if cell["cell_type"] == "code"
        ]
        assert len(code_indexes) == 4
        for index in code_indexes:
            ast.parse("".join(cells[index]["source"]))

        reference_index = next(
            index
            for index, cell in enumerate(cells)
            if cell["cell_type"] == "markdown"
            and "".join(cell["source"]).startswith("## Original Tutorial Reference")
        )
        assert max(code_indexes) < reference_index
        assert all(cell["cell_type"] != "code" for cell in cells[reference_index + 1 :])
        assert len({cell["id"] for cell in cells}) == len(cells)

        active_source = "\n".join(
            "".join(cells[index]["source"]) for index in code_indexes
        )
        assert f"run_notebook_experiment('{slug}', context)" in active_source
        assert "wget " not in active_source
        assert "curl " not in active_source
