执行线程：
- 业务线程-live

任务ID：
- 0623T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `.workflow/tasks/0623T003.md`
- `.workflow/reports/0623T003-business.md`
- `local_live_analysis/hyperliquid_tiny_live_m2_flow_taxonomy_0623T003/**`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Split anti-drift trade-flow taxonomy into `fill_support_touch`, `fill_support_visible_queue_depletion`, `adverse_strict_through`, and `neutral_or_opposite_flow`.
- For buy candidates, `trade.side == A` at `px == bid/limit` is now fill-support touch / visible queue depletion, not adverse pressure. `trade.side == A` at `px < limit` remains `adverse_strict_through`.
- For sell candidates, the taxonomy is symmetric: `trade.side == B` at ask/limit is fill-support touch, and `px > limit` is strict-through adverse.
- Anti-drift pressure blocking now uses strict-through adverse quantity and still requires recent adverse BBO evidence; touch-flow support alone does not block.
- Extended `anti_drift_gate_matrix.csv` and `adverse_flow_state.csv` fields with support/adverse split quantities and counts.
- Added focused tests for stable BBO + sell-at-bid touch support pass, strict-through + adverse BBO block, and mixed touch/opposite-flow pass.
- Generated local non-live taxonomy artifacts under `local_live_analysis/hyperliquid_tiny_live_m2_flow_taxonomy_0623T003/` with three scenarios:
  - `touch_support_pass`: allowed, `fill_support_touch_qty_btc=0.04`, `adverse_strict_through_qty_btc=0`.
  - `strict_through_adverse_bbo_block`: blocked, `adverse_strict_through_qty_btc=0.04`, `adverse_bbo_move=True`.
  - `mixed_flow_no_adverse_bbo_pass`: allowed, support touch plus neutral/opposite flow, no strict-through adverse.

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py -q` -> `16 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py examples/hyperliquid/hyperliquid_tiny_live_m2_public_flow_diagnosis.py` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --help` -> passed
- `git diff --check` -> passed
- Local artifact health: `4` files, `0` empty files.

done：
- T003 completed public-flow taxonomy / anti-drift fill-support split repair only.
- Touch-flow support is no longer misclassified as adverse drift, but touch trades are not sufficient by themselves: candidates must still pass fresh-touch evidence, current BBO, queue, size, post-`open_orders` freshness, immediate guard, anti-drift, and maker-only submit rules.
- No live order window was run for this task, no credentials were read, no remote checkout was refreshed, no final gate was rerun, and no real order endpoint was called by this task.
- This does not complete M2, does not prove stable PnL, and does not authorize M3, taker/crossing, one-tick-back, cap relaxation, or default-on behavior.

blockers：
- 无 task-scoped blocker.
- M2 remains blocked on missing live maker fill / fee / inventory / realized PnL proof.

commit：
- `ac76278`

提交信息：
- `0623 split anti drift flow taxonomy`
