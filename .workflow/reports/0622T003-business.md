```md
执行线程：
- 业务线程-live

任务ID：
- 0622T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0622T003.md`
- `.workflow/reports/0622T003-business.md`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_loop.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py`
- `local_live_analysis/hyperliquid_tiny_live_m2_same_process_watcher_0622T003/**`
- `local_live_analysis/hyperliquid_tiny_live_m2_same_process_watcher_0622T003_rerun_short_iter/**`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Added a same-process remote watcher/live path so the public watcher, selected current candidate, immediate pre-submit guard, private preflight, post-only `Alo` submit/cancel path, pullback, final open-orders proof, and T008 ledger can run from one `awsserver1` process after final gate go.
- Removed the `0622T002` controller-pullback-before-order path for this task by adding `--same-process-live` and passing the exact selected candidate context into `fill_window.run_window(...)`.
- Added an immediate same-process fresh-touch guard that rechecks quote age, current BBO, selected quote at touch, post-only non-crossing state, size `<=0.005 BTC`, queue-depth/order-count quality band, and `Alo` TIF before any submit.
- Added latest-candidate selection so a later allowed row replaces an older allowed row inside the same public precheck iteration.
- Added same-process latency, immediate guard, selected-candidate, no-submit, and window-result artifacts.

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py examples/hyperliquid/test_hyperliquid_tiny_live_m2_public_watcher.py examples/hyperliquid/test_hyperliquid_tiny_live_m2_pnl_ledger.py -q` -> `29 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py examples/hyperliquid/hyperliquid_tiny_live_m2_fill_loop.py` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --help` -> passed and includes `--same-process-live`
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py --help` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_fill_loop.py --help` -> passed
- `git diff --check` -> passed
- Formal same-process controller command:
  `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --output-dir local_live_analysis/hyperliquid_tiny_live_m2_same_process_watcher_0622T003 --watcher-seconds 600 --iteration-seconds 20 --candidate-stride-seconds 1 --max-order-size 0.005 --wait-seconds 10 --quote-hold-seconds 3 --requote-attempts 2`
  -> completed fail-closed with no order submission.
- Formal short-iteration rerun after latest-candidate fix:
  `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --output-dir local_live_analysis/hyperliquid_tiny_live_m2_same_process_watcher_0622T003_rerun_short_iter --watcher-seconds 600 --iteration-seconds 3 --candidate-stride-seconds 1 --max-order-size 0.005 --wait-seconds 10 --quote-hold-seconds 3 --requote-attempts 2`
  -> completed fail-closed with `final_recommendation=hyperliquid_tiny_live_m2_same_process_watcher_blocked`.
- Final artifact health check on `local_live_analysis/hyperliquid_tiny_live_m2_same_process_watcher_0622T003_rerun_short_iter/` -> required files present, `714` files, `0` empty files.
- Redaction scan on final artifacts and task file -> no credential/private key/signature values; matches were only allowed boolean/header fields such as `raw_signatures_written=false`.

done：
- Implementation commits:
  - `1b063a7` / `0622 add same-process M2 watcher live path`
  - `5c58af1` / `0622 tighten same-process M2 trigger freshness`
- Formal final run used remote commit `5c58af1bdd10317ee1953060a37b37d03431188d` after git-safe refresh and final gate go.
- Final gate returned `tiny_live_ready_for_controller_go`, `allow_create_0617T008=true`, and `blocking_reasons=[]`.
- Same-process watcher short-iteration rerun ran `163.946857s` and completed `54` watcher iterations.
- Public counts: `collection_count=54`, `l2Book=81`, `trades=181`, `subscription_ack=108`, `reconnect_count=0`, `total_trade_event_count=1900`, `candidate_count=150`, `eligible_candidate_count=1`.
- Selected candidate was buy `quality_a` at quote `64227`, ask `64228`, size `0.005 BTC`, source same-side top qty `0.00033 BTC`, same-side top order count `2`, source top-depth multiple `0.066`, strict-through qty `0.20921 BTC`, and at-or-through qty `0.20954 BTC`.
- Same-process safety boundary held before order: `public_waiting_phase_private_or_order_endpoint_called=false`, `controller_pullback_before_order=false`, and `separate_live_window_process=false`.
- Immediate pre-submit guard blocked the order: `same_process_guard_status=fail_closed`, reason `trigger_candidate_stale_before_order;selected_quote_not_current_touch;current_top_depth_outside_quality_a_band;current_top_order_count_outside_quality_a_band`.
- Guard facts: candidate age `3.711s` versus max `3.0s`; selected quote `64227`; current BBO had moved to bid/ask `64219/64220`; current same-side top qty `20.52451 BTC`; current same-side top order count `49`; current top-depth multiple `4104.902x`.
- No live order was submitted: `live_submissions_count=0`, `live_window_triggered=false`, `fill_count=0`, `maker_fill_count=0`.
- Independent remote open-orders proof returned `final_open_orders_count=0` and `final_open_orders_empty=true`.
- T008 ledger ran and returned `live_realized_pnl_proof=false`, `realized_pnl_proof_status=fail_closed_no_realized_live_pnl`, and blocker `no_input_windows_found`.
- Artifact paths:
  - preliminary run: `local_live_analysis/hyperliquid_tiny_live_m2_same_process_watcher_0622T003/`
  - final rerun: `local_live_analysis/hyperliquid_tiny_live_m2_same_process_watcher_0622T003_rerun_short_iter/`
- M2 remains blocked. The same-process repair removed the controller pullback / separate live-window latency path, but the current public candidate still decayed before the immediate guard could safely submit.

blockers：
- Immediate guard failed closed because the selected quote was stale, no longer current touch, and current top-of-book queue quality had deteriorated beyond the accepted `quality_a` band.
- No live order was submitted and no live maker fill occurred.
- T008 ledger correctly failed closed with `fail_closed_no_realized_live_pnl`.

commit：
- 1b063a7
- 5c58af1

提交信息：
- 0622 add same-process M2 watcher live path
- 0622 tighten same-process M2 trigger freshness
```
