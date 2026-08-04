#!/usr/bin/env python3
"""Convert the introductory notebooks into runnable Tardis test notebooks."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_raw_cell

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = PROJECT_ROOT / "examples"

NOTEBOOKS = [
    ("getting_started", "Getting Started.ipynb"),
    ("working_with_market_depth_and_trades", "Working with Market Depth and Trades.ipynb"),
    ("data_preparation", "Data Preparation.ipynb"),
    ("fusing_depth_data", "Fusing Depth Data.ipynb"),
    ("order_latency_data", "Order Latency Data.ipynb"),
    ("impact_of_order_latency", "Impact of Order Latency.ipynb"),
    ("accelerated_backtesting", "Accelerated Backtesting.ipynb"),
    ("level_3_backtesting", "Level-3 Backtesting.ipynb"),
    ("integrating_custom_data", "Integrating Custom Data.ipynb"),
    (
        "glft_market_making_model_and_grid_trading",
        "GLFT Market Making Model and Grid Trading.ipynb",
    ),
    ("high_frequency_grid_trading", "High-Frequency Grid Trading.ipynb"),
    (
        "high_frequency_grid_trading_simplified_glft",
        "High-Frequency Grid Trading - Simplified from GLFT.ipynb",
    ),
    (
        "market_making_alpha_order_book_imbalance",
        "Market Making with Alpha - Order Book Imbalance.ipynb",
    ),
    ("market_making_alpha_basis", "Market Making with Alpha - Basis.ipynb"),
    ("market_making_alpha_apt", "Market Making with Alpha - APT.ipynb"),
    ("pricing_framework", "Pricing Framework.ipynb"),
    ("making_multiple_markets", "Making Multiple Markets.ipynb"),
    (
        "making_multiple_markets_introduction",
        "Making Multiple Markets - Introduction.ipynb",
    ),
    ("probability_queue_models", "Probability Queue Models.ipynb"),
    (
        "queue_based_market_making_large_tick",
        "Queue-Based Market Making in Large Tick Size Assets.ipynb",
    ),
    (
        "high_frequency_grid_trading_exchange_comparison",
        "High-Frequency Grid Trading - Comparison Across Other Exchanges.ipynb",
    ),
]

GENERATED_TAG = "tardis-runnable"
REFERENCE_HEADING = "## Original Tutorial Reference"
COPY_ROOT = Path("tutorial_reproduction/notebooks/0804T005")
TASK_BY_SLUG = {}
TARGET_BY_SLUG: dict[str, Path] = {}
for index, (slug, filename) in enumerate(NOTEBOOKS, start=1):
    if index <= 9:
        task_id = "0804T003"
    elif index <= 16:
        task_id = "0804T004"
    else:
        task_id = "0804T005"
    TASK_BY_SLUG[slug] = task_id
    TARGET_BY_SLUG[slug] = (
        COPY_ROOT / filename if task_id == "0804T005" else Path(filename)
    )


def _source(cell: nbformat.NotebookNode) -> str:
    return str(cell.get("source", "")).rstrip()


def _set_cell_id(cell: nbformat.NotebookNode, identity: str) -> nbformat.NotebookNode:
    cell.id = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return cell


def _original_cells(notebook: nbformat.NotebookNode) -> list[nbformat.NotebookNode]:
    if not notebook.metadata.get("tutorial_reproduction", {}).get("generated"):
        return list(notebook.cells)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown" and _source(cell).startswith(REFERENCE_HEADING):
            originals = []
            for reference in notebook.cells[index + 1 :]:
                original_type = reference.metadata.get(
                    "tutorial_reproduction_original_type",
                    reference.cell_type,
                )
                metadata = {
                    key: value
                    for key, value in reference.metadata.items()
                    if key != "tutorial_reproduction_original_type"
                }
                if original_type == "code":
                    originals.append(new_code_cell(_source(reference), metadata=metadata))
                elif original_type == "raw":
                    originals.append(new_raw_cell(_source(reference), metadata=metadata))
                else:
                    originals.append(new_markdown_cell(_source(reference), metadata=metadata))
            original_title = notebook.metadata.get("tutorial_reproduction", {}).get(
                "original_title"
            )
            if original_title:
                originals.insert(0, new_markdown_cell(original_title))
            return originals
    raise ValueError("Generated notebook is missing its original-reference marker")


def _reference_cell(
    cell: nbformat.NotebookNode,
    slug: str,
    index: int,
) -> nbformat.NotebookNode:
    source = _source(cell)
    metadata = {"tutorial_reproduction_original_type": cell.cell_type}
    if cell.cell_type != "code":
        copied = new_markdown_cell(source, metadata=metadata)
        return _set_cell_id(copied, f"{slug}:reference:{index}:{source}")
    return _set_cell_id(
        new_raw_cell(source, metadata=metadata),
        f"{slug}:reference-code:{index}:{source}",
    )


def _active_cells(slug: str, filename: str) -> list[nbformat.NotebookNode]:
    task_id = TASK_BY_SLUG[slug]
    setup = _set_cell_id(
        new_code_cell(
        "\n".join(
            [
                "from pathlib import Path",
                "import sys",
                "",
                "cwd = Path.cwd().resolve()",
                "search_roots = []",
                "for base in (cwd, *cwd.parents):",
                "    search_roots.extend((base, base / 'examples'))",
                "examples_root = next(",
                "    path for path in search_roots",
                "    if (path / 'tutorial_reproduction').is_dir()",
                ")",
                "if str(examples_root) not in sys.path:",
                "    sys.path.insert(0, str(examples_root))",
                "",
                "from tutorial_reproduction.notebook_support import (",
                "    context_dict,",
                "    notebook_context,",
                "    run_notebook_experiment,",
                ")",
            ]
        ),
        metadata={"tags": [GENERATED_TAG]},
        ),
        f"{slug}:active:setup",
    )
    context = _set_cell_id(
        new_code_cell(
        "\n".join(
            [
                f"context = notebook_context({task_id!r})",
                "context_dict(context)",
            ]
        ),
        metadata={"tags": [GENERATED_TAG]},
        ),
        f"{slug}:active:context",
    )
    execute = _set_cell_id(
        new_code_cell(
        "\n".join(
            [
                f"manifest = run_notebook_experiment({slug!r}, context)",
                "manifest['result']",
            ]
        ),
        metadata={"tags": [GENERATED_TAG]},
        ),
        f"{slug}:active:execute",
    )
    verify = _set_cell_id(
        new_code_cell(
        "\n".join(
            [
                "assert manifest['result']['status'] != 'failed'",
                "print('notebook:', manifest['notebook'])",
                "print('status:', manifest['result']['status'])",
                "print('output:', context.output_root)",
            ]
        ),
        metadata={"tags": [GENERATED_TAG]},
        ),
        f"{slug}:active:verify",
    )
    description = _set_cell_id(
        new_markdown_cell(
        "\n".join(
            [
                "## Runnable Tardis Test",
                "",
                "This notebook executes the corresponding experiment through the shared",
                "`tutorial_reproduction` runner. It uses existing Tardis files only and",
                "does not require a Tardis API key or download data.",
                "",
                "Defaults:",
                "",
                "- amdserver: `/home/molly/data/tardis/binance-futures`, `2025-08-01`",
                "- Mac: `~/Documents/tardis`, `2025-01-01`",
                "- Window: `300` seconds",
                "",
                "Optional environment overrides:",
                "",
                "- `HFTBACKTEST_TARDIS_ROOT`",
                "- `HFTBACKTEST_MULTI_TARDIS_ROOT`",
                "- `HFTBACKTEST_TARDIS_DATE`",
                "- `HFTBACKTEST_NOTEBOOK_SECONDS`",
                "- `HFTBACKTEST_NOTEBOOK_OUTPUT`",
                "",
                f"Active experiment: `{filename}` (`{slug}`).",
                (
                    "This file is a generated copy; the original notebook under "
                    "`examples/` remains unchanged."
                    if task_id == "0804T005"
                    else ""
                ),
            ]
        ),
        metadata={"tags": [GENERATED_TAG]},
        ),
        f"{slug}:active:description",
    )
    return [description, setup, context, execute, verify]


def _read_source_notebook(
    path: Path,
    source_ref: str | None,
) -> nbformat.NotebookNode:
    if source_ref is None:
        return nbformat.read(path, as_version=4)
    relative_path = path.relative_to(PROJECT_ROOT)
    source = subprocess.run(
        ["git", "show", f"{source_ref}:{relative_path}"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return nbformat.reads(source, as_version=4)


def refresh_notebook(slug: str, filename: str, source_ref: str | None = None) -> None:
    source_path = EXAMPLES_ROOT / filename
    target_path = EXAMPLES_ROOT / TARGET_BY_SLUG[slug]
    notebook = _read_source_notebook(source_path, source_ref)
    originals = _original_cells(notebook)
    title = originals[0] if originals and originals[0].cell_type == "markdown" else None
    reference_source = [
        REFERENCE_HEADING,
        "",
        "The original tutorial narrative and code are retained below for comparison.",
        "Original code cells are rendered as non-executing references so that",
        "`Run All` remains reproducible with the configured Tardis dataset.",
    ]
    cells: list[nbformat.NotebookNode] = []
    if title is not None:
        cells.append(
            _set_cell_id(new_markdown_cell(_source(title)), f"{slug}:active:title")
        )
        originals = originals[1:]
    cells.extend(_active_cells(slug, filename))
    cells.append(
        _set_cell_id(
            new_markdown_cell("\n".join(reference_source)),
            f"{slug}:reference-heading",
        )
    )
    cells.extend(
        _reference_cell(cell, slug, index)
        for index, cell in enumerate(originals, start=1)
    )

    metadata = dict(notebook.metadata)
    metadata["tutorial_reproduction"] = {
        "generated": True,
        "task_id": TASK_BY_SLUG[slug],
        "slug": slug,
        "original_title": _source(title) if title is not None else "",
        "source_notebook": str(source_path.relative_to(PROJECT_ROOT)),
        "target_notebook": str(target_path.relative_to(PROJECT_ROOT)),
    }
    refreshed = new_notebook(cells=cells, metadata=metadata)
    nbformat.validate(refreshed)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(refreshed, target_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate generated notebook metadata without rewriting files.",
    )
    parser.add_argument(
        "--source-ref",
        help="Read original notebooks from this Git revision before rewriting.",
    )
    parser.add_argument(
        "--task-id",
        choices=sorted(set(TASK_BY_SLUG.values())),
        help="Rewrite or check only notebooks assigned to this task.",
    )
    args = parser.parse_args()
    selected = [
        (slug, filename)
        for slug, filename in NOTEBOOKS
        if args.task_id is None or TASK_BY_SLUG[slug] == args.task_id
    ]
    if args.check:
        for slug, filename in selected:
            notebook = nbformat.read(
                EXAMPLES_ROOT / TARGET_BY_SLUG[slug],
                as_version=4,
            )
            nbformat.validate(notebook)
            metadata = notebook.metadata.get("tutorial_reproduction", {})
            if metadata.get("slug") != slug or not metadata.get("generated"):
                raise ValueError(f"{filename} is not generated for {slug}")
        return
    for slug, filename in selected:
        refresh_notebook(slug, filename, source_ref=args.source_ref)


if __name__ == "__main__":
    main()
