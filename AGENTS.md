# AGENTS.md

This repository uses `workflow-kit` for task dispatch, execution reports, QA acceptance, and progress tracking.

The current active development focus is the Binance maker market-making workflow under:

- `examples/binance_tick_mm/`
- `docs/*plan*.md`
- `docs/binance_tick_mm*.md`
- `docs/maker_optimization_acceptance.md`
- `local_live_analysis*/`
- `baselines/`

## Required Workflow

Before starting a non-trivial task, read:

- `.workflow/workflow-kit/workflow-manual.md`
- `.workflow/workflow-kit/task-dispatch-template.md`
- `.workflow/workflow-kit/thread-report-template.md`
- `.workflow/workflow-kit/qa-acceptance-template.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

## Role Model

- 总控：the main planning session. It defines scope, writes task files, reads QA results, and decides the next task.
- 业务线程：the execution session. It changes code or runs verification within the assigned scope.
- 测试线程：used for focused test discovery, regression runs, and evidence collection.
- QA验收线程：checks whether the task met its acceptance criteria and writes the latest QA result.

Default chain:

```text
业务线程 -> QA验收线程 -> 总控
```

## Task Rules

- Every formal task must have a task file under `.workflow/tasks/`.
- Task IDs use `MMDDTxxx`, for example `0510T001`.
- Use only these statuses:
  - `待执行`
  - `执行中`
  - `待验收`
  - `已通过`
  - `未通过`
  - `阻塞`
  - `作废`
- Default to one formal task at a time unless tasks are independent and explicitly marked as parallel.
- Do not silently expand task scope.
- Do not modify unrelated files.

## Report Rules

- Every execution result must be reported under `.workflow/reports/`.
- Business/test reports should normally end in `待验收`, not `已通过`.
- QA reports may end only in `已通过`, `未通过`, or `阻塞`.
- The latest QA result should also be copied to `docs/qa-acceptance-report.md`.

## Verification Rules

For hftbacktest, choose the smallest useful verification command for the task scope.

For the active Binance maker MM work, start with focused commands around `examples/binance_tick_mm`:

- `python -m pytest examples/binance_tick_mm`
- `python examples/binance_tick_mm/maker_acceptance.py --help`
- `python examples/binance_tick_mm/align_live_run.py --help`
- `python examples/binance_tick_mm/run_env_test.py --help`

For repository-level work:

- Rust workspace or core logic: `cargo test`
- Single crate: `cargo test -p <crate>`
- Python bindings: inspect `py-hftbacktest/README.md`; likely requires `maturin develop` before Python tests
- Focused Python tests: `python -m pytest py-hftbacktest/tests`

If a command cannot run because of dependencies or environment setup, report the blocker instead of inventing a passing result.
