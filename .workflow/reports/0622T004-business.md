执行线程：
- 业务线程-live

任务ID：
- 0622T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_loop.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py`
- `.workflow/tasks/0622T004.md`
- `.workflow/reports/0622T004-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `local_live_analysis/hyperliquid_tiny_live_m2_event_driven_current_candidate_0622T004_rerun_fast_submit/**`

action：
- Implemented event-driven current-candidate watcher/live path that evaluates fresh-touch eligibility from current in-memory L2/BBO plus rolling public trades and calls the live window in the same remote process after immediate guard pass.
- Added `fast_event_driven_submit` to the live fill window so event-driven live mode avoids slow pre-submit `all_mids` / `meta` / `user_state` / `user_fees` reads while preserving pre-submit open-orders guard, current L2 guard, `Alo`, `<=0.005 BTC`, max-loss, tracked cancel, final open-orders proof, and T008 fail-closed.
- First formal run on `59ec906` proved the outer event-driven trigger was fast but exposed inner private-preflight staleness: `candidate_event_to_guard_start=0.00016s`, then inner guard failed at candidate ages `1.032s` / `1.036s` versus the `1.0s` cap before any order.
- Committed the fast-submit repair as `f6487c0` and reran the formal controller into `local_live_analysis/hyperliquid_tiny_live_m2_event_driven_current_candidate_0622T004_rerun_fast_submit/`.
- Formal rerun refreshed `awsserver1:/home/admin/hftbacktest-cross-exchange` from `59ec9069e` to `f6487c063` using a `2786` byte incremental bundle; remote branch was `cross-exchange`, dirty count `0`, SDK availability `true`.
- Final gate returned `tiny_live_ready_for_controller_go`, `allow_create_0617T008=true`, and `blocking_reasons=[]`.
- Event-driven watcher rerun completed after `11.32509s`, with public counts `l2Book=3`, `trades=3`, `subscriptionResponse=2`, expanded trade events `56`, event-driven evaluations `6`, current candidates `6`, and trigger count `1`.
- Outer event guard passed: `candidate_event_to_guard_start=0.000286s`, immediate guard elapsed `0.000024s`, selected buy quote `64122`, size `0.005 BTC`, quality `quality_a`, current bid/ask `64122/64123`, top depth multiple `8.114x`, `Alo`, current-touch match, post-only non-crossing.
- Inner fill-window guard passed after the repair: candidate age `0.609s`, current bid/ask `64107/64108`, selected buy size `0.005 BTC`, quality `quality_a`, top depth multiple `2.684x`, `fast_event_driven_submit=true`.
- One real post-only `Alo` order submission was attempted: buy `0.005 BTC` at `64107.0`, notional `320.535 USDC`.
- Hyperliquid returned order `error`: post-only order would have immediately matched because BBO had moved to `64090@64091`. This preserved maker-only `Alo` behavior and produced no fill.
- Attempt 1 quote-aging guard recorded `lost_touch+adverse_drift`, with BBO moving from `64107/64108` to `64090/64091` after about `1.004622s`; attempt 2 was skipped because current quality bands were no longer eligible.
- Tracked cancel by cloid was attempted, final open orders were empty, shutdown proof was `pass`, and independent remote open-orders check returned `final_open_orders_empty=true`.
- T008 ledger ran on pulled-back artifacts and returned `live_realized_pnl_proof=false`, `realized_pnl_proof_status=fail_closed_no_realized_live_pnl`, `fill_count=0`, `maker_fill_count=0`.

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py -q` -> `3 passed`
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_public_watcher.py -q` -> `4 passed`
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py -q` -> `21 passed`
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_pnl_ledger.py -q` -> `5 passed`
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_public_watcher.py examples/hyperliquid/test_hyperliquid_tiny_live_m2_pnl_ledger.py -q` -> `9 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py examples/hyperliquid/hyperliquid_tiny_live_m2_fill_loop.py` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --help` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py --help` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_fill_loop.py --help` -> passed
- `git diff --check` -> passed
- Formal remote refresh, final gate, live event-driven watcher, pullback, independent open-orders check, and T008 ledger executed successfully.
- Artifact health: `93` files, `0` empty files in the rerun output directory.
- Redaction scan: no raw 64-hex secret material found; matches were expected task-boundary text, boolean manifest fields, or redacted credential-source field names.

done：
- T004 repaired the previous inner pre-submit staleness blocker and reached a real maker-only `Alo` submit attempt under the unchanged `<=0.005 BTC` cap.
- T004 did not complete M2. The live order did not rest or fill because exchange-side post-only validation rejected it after fast BBO drift; T008 therefore failed closed with no live realized PnL proof.
- Current M2 blocker is no live maker fill / fee / inventory / realized PnL proof after a valid event-driven trigger, not controller pullback, separate-window latency, open-orders shutdown, or T008 arithmetic.

blockers：
- M2 remains blocked: `live_trigger_without_t008_realized_pnl_proof`.
- The latest order attempt was rejected by Hyperliquid post-only protection because BBO moved to `64090@64091` before exchange processing, so `fill_count=0` and `maker_fill_count=0`.

commit：
- `59ec906`
- `f6487c0`

提交信息：
- `0622 add event-driven M2 current candidate watcher`
- `0622 trim event-driven M2 pre-submit path`
