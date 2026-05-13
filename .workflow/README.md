# hftbacktest Workflow

This directory contains the persistent workflow state for hftbacktest development.

## Contents

- `workflow-kit/`: workflow manuals and templates.
- `tasks/`: task dispatch files.
- `reports/`: execution reports and QA reports.
- `build_dashboard.py`: local dashboard generator.
- `dashboard.html`: generated task dashboard.
- `dispatch_suggestions.md`: generated next-step suggestions.

## Run Dashboard

From the repository root:

```bash
python3 .workflow/build_dashboard.py
```

Then open:

```text
.workflow/dashboard.html
```

## First Task

The first task is:

```text
0510T001 - 建立 binance_tick_mm live/backtest 闭环任务模板
```

Dispatch it to the testing thread before doing implementation work.
