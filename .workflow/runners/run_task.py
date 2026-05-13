#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER_DIR = Path(__file__).resolve().parent


def load_runner(task_id: str):
    runner_path = RUNNER_DIR / f"run_{task_id}.py"
    if not runner_path.exists():
        raise SystemExit(f"No runner found for task {task_id}: {runner_path}")

    spec = importlib.util.spec_from_file_location(f"workflow_runner_{task_id}", runner_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load runner: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run"):
        raise SystemExit(f"Runner {runner_path} does not define run(repo_root: Path) -> int")
    return module.run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a workflow task runner")
    parser.add_argument("task_id", help="Task ID, for example 0510T002")
    args = parser.parse_args()

    task_id = args.task_id.upper()
    run = load_runner(task_id)
    return int(run(ROOT) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
