执行线程：
- 业务线程-local

任务ID：
- 0707T006

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `.workflow/tasks/0707T006.md`
- `.workflow/reports/0707T006-business.md`

sync：
- Before task execution, local `cross-exchange`, GitHub `origin/cross-exchange`, and `amdserver:~/workspace/hftbacktest` were aligned to `af3bb1ac38783eb18e04da1369dc95eafd3f5f95`.

artifact：
- `local_live_analysis/cross_exchange_t010_inline_reprice_handoff_contract_repair_0707T006/`

action：
- Added explicit handoff/instrumentation fields to `immediate_fresh_touch_guard` output:
  - `handoff_phase`
  - trigger candidate source/age/side/quote/size/quality/freshness/skip fields
  - current reprice allowed/skip/source/age fields
- Updated the event-driven inline reprice path to pass the original trigger candidate separately from the post-open-orders current reprice decision.
- Added `post_open_orders_handoff_latency_exceeded` as a specific fail-closed reason for post-open-orders inline reprice when the original trigger candidate is already older than the immediate guard max age.
- Suppressed ambiguous missing intent / missing quality bucket reason mixtures when the post-open-orders handoff latency gate is already the decisive failure.
- Updated immediate guard CSV fieldnames so artifacts preserve both trigger-candidate and current-reprice evidence.
- Added focused pytest coverage for stale handoff preservation and updated existing stale-guard expectations.

boundary：
- No anti-drift threshold change.
- No touch-stability threshold change.
- No edge threshold change.
- No quote-envelope change.
- No order size change.
- No max-submission change.
- No open-orders/L2-resync latency optimization.
- No live-submit.

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py -q` passed: `42 passed`.
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py` passed.
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --help` passed.
- Local artifact generation passed; summary shows `guard_reason=post_open_orders_handoff_latency_exceeded`, `trigger_candidate_quality_bucket=quality_a`, and `current_reprice_skip_reason=outside_quality_a_b_queue_bands`.
- `git diff --check` passed.

done：
- Handoff contract/instrumentation repair completed locally and is ready for QA.
- Full `0625T010` remains blocked pending QA and a separately authorized controlled live evidence rerun after this repair.

blockers：
- No blockers for this repair task.
- Full `0625T010` remains blocked because this task did not collect submitted lifecycle, fill/no-fill economics, fee/rebate, inventory transition, realized PnL, or replay-ready live order evidence.

commit：
- pending

提交信息：
- pending
