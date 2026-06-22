执行线程：
- 业务线程-live

任务ID：
- 0622T005

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`
- `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_loop.py`
- `examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py`
- `.workflow/tasks/0622T005.md`
- `.workflow/reports/0622T005-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `local_live_analysis/hyperliquid_tiny_live_m2_inline_reprice_0622T005/**`

action：
- Implemented watcher-local inline reprice / submit path behind `--event-driven-inline-reprice-live`, so the triggered submit path no longer calls the full `fill_window.run_window` path after a public candidate trigger.
- Kept the waiting phase public-only, then after trigger used only private `open_orders` as the pre-submit safety check, repriced from latest in-memory BBO/current L2, ran strict immediate guard, and submitted Hyperliquid post-only `Alo`.
- Added maker-only post-only reject handling: on would-immediately-match reject, wait for the next public event / latest in-memory BBO, re-evaluate fresh-touch / dynamic-size / session-gate, and allow at most one retry while total real order endpoint calls remain capped at `2`.
- Added inline artifacts: `inline_reprice_manifest.json`, `inline_reprice_latency_matrix.csv`, `inline_reprice_attempt_matrix.csv`, `inline_reprice_guard_matrix.csv`, and `inline_reprice_post_only_reject_matrix.csv`.
- Preserved tracked cancel, final open-orders proof, independent remote open-orders check, redaction, and T008 ledger fail-closed.
- Formal remote run refreshed `awsserver1:/home/admin/hftbacktest-cross-exchange` from `f6487c063e0d895c6bc118f6a1619d1cfbb855fb` to `c4faf36b7a60342f195238041d2b711ca300233e` using a `23139` byte incremental bundle; remote branch was `cross-exchange`, dirty count `0`, SDK availability `true`, and Python was `/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python` at `Python 3.13.5`.
- Final gate returned `tiny_live_ready_for_controller_go`, `allow_create_0617T008=true`, and `blocking_reasons=[]`.
- Event-driven inline watcher ran `82.208056s` of the `600s` timebox, collected public counts `l2Book=16`, `trades=32`, `subscriptionResponse=2`, `pong=2`, expanded to `93` trade events, evaluated `48` current candidates, and triggered once.
- The public waiting phase stayed public-only before trigger; private/order endpoints were not called until the trigger path.
- Inline attempt 1 passed guard with current/submit BBO `64144/64145`, candidate age `0.559s`, buy quote `64144.0`, size `0.00004 BTC`, current same-side top qty `0.00017 BTC`, top order count `1`, top-depth multiple `4.25x`, and `quality_a`.
- Attempt 1 latency split: `trigger_to_open_orders_start=0.000255s`, `open_orders_elapsed=0.2836s`, `open_orders_end_to_reprice=0.000136s`, `reprice_to_order_submit=0.000045s`, and exchange order response `0.358896s`.
- Attempt 1 was rejected by Hyperliquid post-only protection: `Post only order would have immediately matched, bbo was 64143@64144. asset=0`; retry decision was `wait_next_public_event_reprice`.
- Inline attempt 2 waited for the next public event, passed guard with current/submit BBO `64144/64145`, candidate age `0.826s`, buy quote `64144.0`, size `0.00004 BTC`, current same-side top qty `0.00017 BTC`, top order count `1`, top-depth multiple `4.25x`, and `quality_a`.
- Attempt 2 latency split: `trigger_to_open_orders_start=0.000193s`, `open_orders_elapsed=0.04255s`, `open_orders_end_to_reprice=0.000084s`, `reprice_to_order_submit=0.000043s`, and exchange order response `0.413227s`.
- Attempt 2 was rejected by Hyperliquid post-only protection: `Post only order would have immediately matched, bbo was 64142@64143. asset=0`; retry cap was reached and no further order was submitted.
- Live submissions count was `2`; both used `Alo`, buy side, and `0.00004 BTC`, which is far below the unchanged `<=0.005 BTC` hard cap.
- No order rested or filled: `fill_count=0`, `maker_fill_count=0`, `ledger_fill_rows=0`.
- Tracked cancel by cloid was attempted for both tracked refs; final open orders were empty and shutdown proof was `pass`.
- Independent remote open-orders check returned `final_open_orders_empty=true`, `final_open_orders_count=0`, with no order or cancel endpoint call in that independent check.
- T008 ledger ran on pulled-back artifacts and returned `live_realized_pnl_proof=false`, `realized_pnl_proof_status=fail_closed_no_realized_live_pnl`, and all PnL summary rows at zero.

verify：
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py -q` -> `5 passed`
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_public_watcher.py -q` -> `4 passed`
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_fill_loop.py examples/hyperliquid/test_hyperliquid_tiny_live_m2_pnl_ledger.py -q` -> `26 passed`
- `python -m py_compile examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py examples/hyperliquid/hyperliquid_tiny_live_m2_fill_loop.py` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py --help` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py --help` -> passed
- `python examples/hyperliquid/hyperliquid_tiny_live_m2_fill_loop.py --help` -> passed
- `python -m pytest examples/hyperliquid/test_hyperliquid_tiny_live_m2_event_driven_watcher.py examples/hyperliquid/test_hyperliquid_tiny_live_m2_public_watcher.py -q` -> `9 passed`
- `git diff --check` -> passed before live execution.
- Formal remote refresh, final gate, live inline watcher, pullback, independent open-orders check, and T008 ledger executed successfully.
- Artifact health: `112` files and `0` empty files under `local_live_analysis/hyperliquid_tiny_live_m2_inline_reprice_0622T005/`.
- Redaction scan found no raw secrets/private keys/signatures; only expected ledger sha256 manifest lines matched 64-hex patterns.

done：
- T005 removed the remaining `watcher -> fill_window` submit overhead and proved the local inline path is effectively immediate after reprice: `open_orders_end_to_reprice` was about `0.084ms-0.136ms`, and `reprice_to_order_submit` was about `0.043ms-0.045ms`.
- T005 did not complete M2. Both valid maker-only `Alo` submit attempts were rejected by exchange-side post-only validation after sub-second BBO drift; no live maker fill, fee, inventory transition, or realized PnL proof exists.
- Current M2 blocker remains `live_trigger_without_t008_realized_pnl_proof`.
- Artifacts are under `local_live_analysis/hyperliquid_tiny_live_m2_inline_reprice_0622T005/`.

blockers：
- M2 remains blocked because T008 returned `fail_closed_no_realized_live_pnl`.
- The actionable bottleneck has narrowed: after inline reprice, local reprice-to-submit latency is tiny; the remaining observed gap is exchange-side BBO drift between local submit intent and Hyperliquid post-only validation, with private `open_orders` still contributing `0.04255s-0.2836s` before the final reprice step.

commit：
- `c74bad3`
- `741b5b2`
- `c4faf36`

提交信息：
- `0622 create inline reprice M2 task`
- `0622 add inline reprice M2 watcher path`
- `0622 defer inline private pullbacks until submit`
