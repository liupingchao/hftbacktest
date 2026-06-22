执行线程：
- 业务线程-live

任务ID：
- 0623T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `.workflow/tasks/0623T001.md`
- `.workflow/reports/0623T001-business.md`
- `local_live_analysis/hyperliquid_tiny_live_m2_state_freshness_0623T001/**`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Added post-`open_orders()` public L2 freshness gating to the watcher-local inline reprice path.
- Added public-state metadata on `EventDrivenPublicState`: `public_state_seq`, `current_public_state_seq`, `current_public_state_channel`, `current_public_state_exchange_time_ms`, `current_public_state_local_receive_ts_ns`, `current_l2_state_seq`, and `current_l2_local_receive_ts_ns`.
- Added `observe_post_open_orders_l2_state()` so submit repricing can continue only after a new `l2Book` is observed with local receive time strictly after `open_orders_end_ns`.
- Added a bounded `0.2s` fail-closed wait. If the source exhausts, disconnects, only times out, or no post-open L2 arrives, the path records `post_open_orders_public_state_stale` / precise reason and does not call the order endpoint.
- Added `public_state_freshness_matrix.csv`, public-state seq columns in latency / attempt matrices, and manifest counts for post-open pass/block evidence.
- Updated inline/anti-drift tests to provide a post-open L2 in pass scenarios and added a stale post-open negative test that proves no order intent is created.
- Generated local non-live evidence under `local_live_analysis/hyperliquid_tiny_live_m2_state_freshness_0623T001/`:
  - `post_open_orders_l2_pass`: post-open L2 observed, `post_open_orders_public_state_pass_count=1`, mock `live_submissions_count=1`.
  - `post_open_orders_l2_block`: no post-open L2 observed, `post_open_orders_public_state_block_count=1`, mock `live_submissions_count=0`, order endpoint not called.

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py -q` -> `11 passed`
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_public_watcher.py -q` -> `4 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --help` -> passed
- `git diff --check` -> passed
- Local artifact smoke check:
  - pass case: `live_submissions_count=1`, `post_open_orders_public_state_pass_count=1`
  - block case: `live_submissions_count=0`, `post_open_orders_public_state_block_count=1`, mock order intents `0`

done：
- T001 completed the state-freshness repair only.
- The inline submit path now reprices only from an L2 snapshot observed after private `open_orders()` returns, or fails closed before order submission.
- No live order window was run for this task, no credentials were read, no remote checkout was refreshed, no final gate was rerun, and no real order endpoint was called by this task.
- This does not complete M2, does not prove stable PnL, and does not authorize M3, taker/crossing, one-tick-back, cap relaxation, or default-on behavior.

blockers：
- 无 task-scoped blocker.
- M2 remains blocked on missing live maker fill / fee / inventory / realized PnL proof.

commit：
- `fb63066`

提交信息：
- `0623 add post open_orders state freshness gate`
