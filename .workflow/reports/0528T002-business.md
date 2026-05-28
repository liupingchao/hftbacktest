```md
执行线程：
- 业务线程-python

任务ID：
- 0528T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0528T002.md`
- `.workflow/reports/0528T002-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `.workflow/dashboard.html`
- `.workflow/dispatch_suggestions.md`
- `examples/binance_tick_mm/backtest_tick_mm.py`
- `examples/binance_tick_mm/test_backtest_tick_mm.py`
- `examples/binance_tick_mm/execution_outcome_calibration.py`
- `examples/binance_tick_mm/test_execution_outcome_calibration.py`
- `examples/binance_tick_mm/align_live_run.py`
- `examples/binance_tick_mm/test_align_live_run.py`
- `local_live_analysis/stage6_audit_bloat_repair_0528T002/run_manifest.json`
- `local_live_analysis/stage6_audit_bloat_repair_0528T002/compact_lifecycle_sample_summary.json`
- `local_live_analysis/stage6_audit_bloat_repair_0528T002/compact_lifecycle_full_prefix_summary.json`
- `local_live_analysis/stage6_audit_bloat_repair_0528T002/stage6_validation/**`

action：
- Implemented `CompactLifecycleAuditWriter` for audit replay compact lifecycle output.
- In audit replay mode, compact lifecycle export is enabled by default unless explicitly disabled by config.
- Compact export preserves decision rows and non-terminal lifecycle rows, and de-duplicates repeated terminal lifecycle rows by first `event_type + order_id`.
- Wired `align_live_run.py` audit replay configs to write `audit_bt_audit_replay.compact_lifecycle.csv`.
- Wired Stage 6 `execution_outcome_calibration.py` to prefer `out/backtest_audit_replay/audit_bt_audit_replay.compact_lifecycle.csv` and fall back to legacy `audit_bt_audit_replay.csv`.
- Added regression tests for terminal de-dup, non-terminal preservation, compact artifact preference, and audit replay config output path.
- Updated workflow status, plan/progress/findings, and dashboard.

verify：
- `python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -q` -> passed, `141 passed`.
- `python -m pytest examples/binance_tick_mm/test_execution_outcome_calibration.py -q` -> passed, `5 passed`.
- `python -m pytest examples/binance_tick_mm/test_align_live_run.py -q` -> passed, `9 passed`.
- Bounded `0526T008` full-audit prefix reproduction:
  - source: `local_live_analysis/5-26-active-minmove-control-30min-b/out/backtest_audit_replay/audit_bt_audit_replay.full.csv`
  - scanned rows: `250000`
  - scanned `cancel_ack` rows: `227080`
  - compact rows written: `23345`
  - duplicate terminal rows skipped: `226655`
- Stage 6 contract validation:
  - command: `python examples/binance_tick_mm/execution_outcome_calibration.py --run-dir local_live_analysis/stage6_audit_bloat_repair_0528T002/run_dir --output-dir local_live_analysis/stage6_audit_bloat_repair_0528T002/stage6_validation`
  - result: passed, `decision_state=methodology_valid_single_sample`
  - row counts: `live_submit_orders=11089`, `replay_submit_orders=11084`, `matched_submit_orders=11084`, `live_filled_orders=471`, `replay_filled_orders=480`
- `git diff --check` -> passed.
- `python3 .workflow/build_dashboard.py` -> passed.

done：
- The formal compact lifecycle artifact contract is `out/backtest_audit_replay/audit_bt_audit_replay.compact_lifecycle.csv`.
- Stage 6 no longer depends on manual lifecycle-min renaming when the compact artifact exists.
- The bounded reproduction demonstrates the original 21GB-style `cancel_ack` bloat is removed by terminal lifecycle de-dup within the compact artifact.
- The original `audit_bt_audit_replay.full.csv` forensic evidence was not deleted or overwritten.
- No strategy behavior, live behavior, fill/cancel replay semantics, queue/touch logic, parameters, guards, default-on behavior, live run, parameter search, tiny-live, or promotion claim was changed.

blockers：
- 无

commit：
- 待提交

提交信息：
- 待提交
```
