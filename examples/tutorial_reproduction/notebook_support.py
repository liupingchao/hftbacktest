"""Shared configuration and execution helpers for the tutorial notebooks."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .run import (
    PreparedData,
    _experiment_accelerated,
    _experiment_custom_data,
    _experiment_data_preparation,
    _experiment_depth_and_trades,
    _experiment_fusing,
    _experiment_getting_started,
    _experiment_latency_impact,
    _experiment_level3,
    _experiment_order_latency,
    _prepare_data,
    _run_experiment,
    _write_json,
)
from .advanced_experiments import (
    experiment_apt,
    experiment_basis,
    experiment_glft,
    experiment_high_frequency_grid,
    experiment_obi,
    experiment_pricing_framework,
    experiment_simplified_glft,
)

EXPERIMENTS: list[tuple[str, str, Callable[[PreparedData, Path], dict[str, Any]]]] = [
    ("getting_started", "Getting Started.ipynb", _experiment_getting_started),
    (
        "working_with_market_depth_and_trades",
        "Working with Market Depth and Trades.ipynb",
        _experiment_depth_and_trades,
    ),
    ("data_preparation", "Data Preparation.ipynb", _experiment_data_preparation),
    ("fusing_depth_data", "Fusing Depth Data.ipynb", _experiment_fusing),
    ("order_latency_data", "Order Latency Data.ipynb", _experiment_order_latency),
    (
        "impact_of_order_latency",
        "Impact of Order Latency.ipynb",
        _experiment_latency_impact,
    ),
    ("accelerated_backtesting", "Accelerated Backtesting.ipynb", _experiment_accelerated),
    ("level_3_backtesting", "Level-3 Backtesting.ipynb", _experiment_level3),
    ("integrating_custom_data", "Integrating Custom Data.ipynb", _experiment_custom_data),
    (
        "glft_market_making_model_and_grid_trading",
        "GLFT Market Making Model and Grid Trading.ipynb",
        experiment_glft,
    ),
    (
        "high_frequency_grid_trading",
        "High-Frequency Grid Trading.ipynb",
        experiment_high_frequency_grid,
    ),
    (
        "high_frequency_grid_trading_simplified_glft",
        "High-Frequency Grid Trading - Simplified from GLFT.ipynb",
        experiment_simplified_glft,
    ),
    (
        "market_making_alpha_order_book_imbalance",
        "Market Making with Alpha - Order Book Imbalance.ipynb",
        experiment_obi,
    ),
    (
        "market_making_alpha_basis",
        "Market Making with Alpha - Basis.ipynb",
        experiment_basis,
    ),
    (
        "market_making_alpha_apt",
        "Market Making with Alpha - APT.ipynb",
        experiment_apt,
    ),
    (
        "pricing_framework",
        "Pricing Framework.ipynb",
        experiment_pricing_framework,
    ),
]

_EXPERIMENT_BY_SLUG = {
    slug: (index, notebook, callback)
    for index, (slug, notebook, callback) in enumerate(EXPERIMENTS, start=1)
}

_TASK_BY_SLUG = {
    slug: ("0804T003" if index <= 9 else "0804T004")
    for index, (slug, _, _) in enumerate(EXPERIMENTS, start=1)
}


@dataclass(frozen=True)
class NotebookContext:
    project_root: str
    tardis_root: str
    date: str
    duration_seconds: int
    output_root: str


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_tardis_root() -> Path:
    candidates = [
        Path("/home/molly/data/tardis/binance-futures"),
        Path("~/Documents/tardis").expanduser(),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    rendered = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "No default Tardis root was found. Set HFTBACKTEST_TARDIS_ROOT.\n"
        f"Checked:\n{rendered}"
    )


def _default_date(tardis_root: Path) -> str:
    if (tardis_root / "trades" / "2025" / "08" / "01" / "BTCUSDT.csv.zst").is_file():
        return "2025-08-01"
    return "2025-01-01"


def notebook_context(task_id: str = "0804T003") -> NotebookContext:
    project_root = _project_root()
    configured_root = os.environ.get("HFTBACKTEST_TARDIS_ROOT")
    tardis_root = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else _default_tardis_root()
    )
    date = os.environ.get("HFTBACKTEST_TARDIS_DATE", _default_date(tardis_root))
    duration_seconds = int(os.environ.get("HFTBACKTEST_NOTEBOOK_SECONDS", "300"))
    if duration_seconds < 30:
        raise ValueError("HFTBACKTEST_NOTEBOOK_SECONDS must be at least 30")
    configured_output = os.environ.get("HFTBACKTEST_NOTEBOOK_OUTPUT")
    output_root = (
        Path(configured_output).expanduser().resolve()
        if configured_output
        else project_root
        / "local_live_analysis"
        / f"tutorial_reproduction_{task_id}"
        / f"{date.replace('-', '')}_{duration_seconds}s"
    )
    return NotebookContext(
        project_root=str(project_root),
        tardis_root=str(tardis_root),
        date=date,
        duration_seconds=duration_seconds,
        output_root=str(output_root),
    )


def context_dict(context: NotebookContext) -> dict[str, Any]:
    return asdict(context)


def prepare_notebook_data(context: NotebookContext | None = None) -> PreparedData:
    context = context or notebook_context()
    return _prepare_data(
        tardis_root=Path(context.tardis_root),
        date=context.date,
        duration_seconds=context.duration_seconds,
        output_root=Path(context.output_root),
    )


def run_notebook_experiment(
    slug: str,
    context: NotebookContext | None = None,
) -> dict[str, Any]:
    context = context or notebook_context()
    if slug not in _EXPERIMENT_BY_SLUG:
        choices = ", ".join(_EXPERIMENT_BY_SLUG)
        raise KeyError(f"Unknown tutorial experiment {slug!r}. Expected one of: {choices}")

    index, notebook, callback = _EXPERIMENT_BY_SLUG[slug]
    prepared = prepare_notebook_data(context)
    output_root = Path(context.output_root)
    result = _run_experiment(
        index,
        slug,
        callback,
        prepared,
        output_root / "experiments",
    )
    manifest = {
        "task_id": _TASK_BY_SLUG[slug],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "notebook": notebook,
        "slug": slug,
        "context": asdict(context),
        "prepared_data": asdict(prepared),
        "result": result,
    }
    _write_json(output_root / "notebook_manifests" / f"{slug}.json", manifest)
    return manifest


def load_notebook_manifest(
    slug: str,
    context: NotebookContext | None = None,
) -> dict[str, Any]:
    context = context or notebook_context()
    path = Path(context.output_root) / "notebook_manifests" / f"{slug}.json"
    return json.loads(path.read_text())
