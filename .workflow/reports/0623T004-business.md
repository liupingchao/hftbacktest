执行线程：
- 业务线程-live

任务ID：
- 0623T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `.workflow/tasks/0623T004.md`
- `.workflow/reports/0623T004-business.md`
- `local_live_analysis/hyperliquid_tiny_live_m2_edge_gate_0623T004/**`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Reviewed accepted read-only pricing-signal / optimistic proxy evidence from `0617T005`, `0617T006`, and canonical pricing-signal artifacts.
- Found no accepted live-compatible decision-time fair-mid provider for the watcher submit path. The accepted artifacts remain read-only / proxy evidence and are not sufficient to be consumed as a real-time edge source.
- Added explicit fair-value edge gate schema and evaluator:
  - `fair_mid_px`
  - `quote_px`
  - `edge_ticks`
  - `signal_age_ms`
  - `fee_buffer_ticks`
  - `adverse_selection_buffer_ticks`
  - `edge_gate_status`
  - `edge_gate_reason`
- Added `edge_gate=True` / `edge_signal_provider` injection to `run_event_driven_inline_reprice_live`.
- Added CLI flag `--event-driven-edge-gate-live`. It enables the fail-closed adapter only; without a live-compatible provider it records `edge_signal_missing_live_compatible_source` and does not invent edge from offline CSV artifacts.
- Placed the gate after post-`open_orders()` public L2 freshness, inline fresh-touch/current BBO guard, and anti-drift, but before `OrderIntent` submission / `executor.run_order_once`.
- Edge rule:
  - buy requires `(fair_mid_px - quote_px) / tick_size > fee_buffer_ticks + adverse_selection_buffer_ticks`;
  - sell requires `(quote_px - fair_mid_px) / tick_size > fee_buffer_ticks + adverse_selection_buffer_ticks`.
- Default T004 parameters:
  - `max_signal_age_ms=250`
  - `required_horizon_ms=1000`
  - `fee_buffer_ticks=2.0`
  - `adverse_selection_buffer_ticks=5.0`
  - `required_edge_ticks=7.0`
- Fail-closed reasons cover missing source, provider exception, missing symbol, wrong symbol, missing horizon, wrong horizon, missing timestamp, future timestamp, stale signal, missing fair mid, invalid quote/tick, invalid side, and insufficient edge.
- Added `edge_gate_matrix.csv`, `edge_gate_manifest.json`, and `edge_gate_no_submit_report.md` outputs.
- Extended `inline_reprice_attempt_matrix.csv` with edge fields so skipped/submitted attempts carry fair-mid / edge / signal freshness evidence.
- Added focused tests for positive edge pass, missing live source fail-closed, stale signal fail-closed, insufficient edge fail-closed, wrong symbol fail-closed, and wrong horizon fail-closed.
- Generated local non-live artifacts under `local_live_analysis/hyperliquid_tiny_live_m2_edge_gate_0623T004/` with four mock scenarios:
  - `positive_edge_pass`: injected provider, `edge_gate_pass_count=1`, `live_submissions_count=1`, mock client order calls `1`.
  - `missing_live_source_block`: no provider, `edge_gate_block_count=1`, `live_submissions_count=0`, mock client order calls `0`.
  - `stale_signal_block`: injected stale provider, `edge_gate_block_count=1`, `live_submissions_count=0`, mock client order calls `0`.
  - `insufficient_edge_block`: injected below-buffer provider, `edge_gate_block_count=1`, `live_submissions_count=0`, mock client order calls `0`.

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py -q` -> `21 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --help` -> passed
- `git diff --check` -> passed
- Local artifact generation for `local_live_analysis/hyperliquid_tiny_live_m2_edge_gate_0623T004/` -> passed

done：
- T004 completed fair-value edge gate integration as an additional pre-submit requirement for the watcher-local inline reprice path.
- Positive edge can pass and reach the existing mock maker-only `Alo` order path.
- Missing, stale, wrong-symbol, wrong-horizon, and below-buffer edge fail closed before order submission.
- The current blocker is explicitly recorded: no accepted live-compatible fair-mid provider exists yet for production/live decision time.
- No live order window was run for this task, no credentials were read, no remote checkout was refreshed, no final gate was rerun, and no real order endpoint was called by this task.
- This does not complete M2, does not prove live maker fill, fee/inventory accounting, realized PnL, M3 readiness, maker viability, or stable PnL.
- This does not authorize taker/crossing, `Ioc`, one-tick-back, cap relaxation, default-on behavior, live execution, or promotion.

blockers：
- 无 task-scoped implementation blocker.
- Live-compatible fair-mid source remains missing and is deliberately represented by a fail-closed adapter / source blocker.
- M2 remains blocked on missing live maker fill / fee / inventory / realized PnL proof.

commit：
- `96fe3c0`

提交信息：
- `0623 add fair value edge gate`
