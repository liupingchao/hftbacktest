# Thread Playbook

This playbook adapts `workflow-kit` to hftbacktest.

## Default Chain

```text
业务线程 / 测试线程 -> QA验收线程 -> 总控
```

## Total Controller

The controller must:

- Create one task file under `.workflow/tasks/`.
- Keep scope, files, action, verification, and QA mode explicit.
- Read `.workflow/reports/*-qa.md` before deciding the next task.
- Update `progress.md` and `findings.md` after completed work.

## Business Threads

Business threads must:

- Stay inside the assigned files/modules.
- Run the requested verification or explain why it cannot run.
- Write one report under `.workflow/reports/`.
- Explicitly include `是否进行QA验收`.

## Testing Thread

The testing thread is responsible for:

- Discovering test commands.
- Running focused or full tests when appropriate.
- Recording exact commands, exit codes, failures, and blockers.
- Avoiding business code changes unless explicitly assigned.

## QA Acceptance Thread

The QA thread must:

- Read the task file and latest execution report.
- Judge only the assigned acceptance criteria.
- Write `.workflow/reports/<task-id>-qa.md`.
- Copy the latest valid QA result to `docs/qa-acceptance-report.md`.

## hftbacktest Verification Hints

Start with the smallest command that proves the task:

- `python -m pytest examples/binance_tick_mm` for the active maker MM example tests.
- `python examples/binance_tick_mm/maker_acceptance.py --help` to verify the acceptance gate script is importable.
- `python examples/binance_tick_mm/align_live_run.py --help` to verify the live/backtest alignment entrypoint is importable.
- `python examples/binance_tick_mm/run_env_test.py --help` to verify the local/live run entrypoint is importable.
- `cargo test` for full Rust workspace verification.
- `cargo test -p hftbacktest` for the core crate.
- `cargo test -p connector` for connector changes.
- `cargo test -p collector` for collector changes.
- `python -m pytest py-hftbacktest/tests` for Python tests after the Python extension is built.
- `maturin develop --manifest-path py-hftbacktest/Cargo.toml` when Python bindings need to be built locally.

If verification is too expensive, blocked, or requires missing dependencies, record that fact in the report.

## Active Binance Maker MM Context

Treat these files as current project context:

- `examples/binance_tick_mm/README.md`
- `docs/5-8-future-plan.md`
- `docs/binance_tick_mm_alignment_execution_plan.md`
- `docs/maker_optimization_acceptance.md`
- `docs/5-4-plan.md`

Do not overwrite historical plans. If new workflow conclusions supersede them, record that in `progress.md` or `findings.md`.

## Standard Live/Backtest Loop

For `examples/binance_tick_mm`, use this loop as the default lifecycle:

1. 采集 live 样本
   - 固定策略参数，不开新规则。
   - Use a clear run id, for example `5-10-day-control-1h-06`.
   - Run about `1H`.
   - Preserve live audit, collector raw gzip, connector logs, and bot logs.
2. 拉回并归档
   - Pull artifacts from `awsserver1` into `local_live_analysis/<run_id>/`.
   - Create `local_live_analysis/archive/<run_id>.tar.gz`.
   - Preserve start/stop times so names do not mislead later analysis.
3. replay/acceptance 验收
   - Run normal replay and audit replay from raw market data.
   - Run `maker_acceptance.py`.
   - First confirm action/planned/reject/throttle are `1.0`.
   - Confirm working semantic/blocking mismatch is `0`.
   - Confirm API/throttle mismatch is `0`.
   - Confirm strict replay lag post-startup breach/drop/fail are `0`.
4. 风险诊断
   - Run `analyze_cancel_fill_risk.py`.
   - Check cancel-fill count, notional rate, and source-path.
   - Separate add-side / same-side readd, inventory worsening without readd, and adverse-selection cancel race.
   - Check markout and cancel-to-fill latency buckets.
5. 判断问题类型
   - If replay/working/API is not aligned, fix the framework first.
   - If alignment passes but cancel-fill risk is high, enter rule design.
   - Record whether the sample points to adverse-selection, same-side readd, or another source-path.
6. 离线规则 replay
   - Run Stage 6J replay on the same samples.
   - Compare baseline, add-side guard, cooldown, and later adverse-selection timing rules.
   - Confirm improvement comes from the intended source-path, not just reduced trading.
7. 跨样本验证
   - Use at least daytime and night-active samples.
   - Do not ship a rule because it works in one window.
   - Compare PnL, max position, drop rate, churn, and cancel-fill source-path stability.
8. live micro test 决策
   - Only enable the rule in small-notional live after multi-sample replay passes.
   - After live, return to step 1.
