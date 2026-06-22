# Progress

## 0622T004 Execution Update

- `0622T004` business execution is complete and is now `待验收`.
- Implementation commits: `59ec906` (`0622 add event-driven M2 current candidate watcher`) and `f6487c0` (`0622 trim event-driven M2 pre-submit path`).
- The task converted the same-process watcher path into an event-driven current-candidate flow using current in-memory L2/BBO plus rolling public trades, and added a live fast-submit path that defers slow `all_mids` / `meta` / `user_state` / `user_fees` pre-submit reads while preserving pre-submit open-orders guard, current L2 guard, `Alo`, `<=0.005 BTC`, tracked cancel, final open-orders proof, and T008 fail-closed.
- Focused local verification passed: event-driven watcher tests `3 passed`, public watcher tests `4 passed`, fill-loop tests `21 passed`, PnL ledger tests `5 passed`, combined public watcher / ledger tests `9 passed`, `py_compile` passed, three CLI help checks passed, and `git diff --check` passed.
- First formal run on `59ec906` proved the outer event-driven trigger path was fast (`candidate_event_to_guard_start=0.00016s`) but found a new blocker: inner `fill_window` private/account preflight pushed candidate age past the `1.0s` guard (`1.032s` / `1.036s`), so no order was submitted.
- The fast-submit repair was committed as `f6487c0` and rerun into `local_live_analysis/hyperliquid_tiny_live_m2_event_driven_current_candidate_0622T004_rerun_fast_submit/`.
- Formal rerun refreshed `awsserver1:/home/admin/hftbacktest-cross-exchange` from `59ec9069e` to `f6487c063` using a `2786` byte incremental bundle; remote branch remained `cross-exchange`, dirty count `0`, Hyperliquid SDK availability `true`, and final gate returned `tiny_live_ready_for_controller_go` with `allow_create_0617T008=true`.
- Event-driven watcher rerun completed after `11.32509s`, saw `l2Book=3`, `trades=3`, `subscriptionResponse=2`, expanded to `56` trade events, evaluated `6` current candidates, and triggered once.
- Outer event guard passed with `candidate_event_to_guard_start=0.000286s`, current bid/ask `64122/64123`, selected buy size `0.005 BTC`, quality `quality_a`, top depth multiple `8.114x`, `Alo`, current-touch match, and post-only non-crossing proof.
- The repaired inner `fill_window` guard also passed before submit with candidate age `0.609s`, current bid/ask `64107/64108`, selected size `0.005 BTC`, top depth multiple `2.684x`, and `fast_event_driven_submit=true`.
- One live post-only `Alo` order submission was attempted at buy `64107.0` for `0.005 BTC`; the exchange returned an order `error` because the post-only order would have immediately matched after BBO moved to `64090@64091`. This is a correct fail-closed `Alo` rejection, not a fill.
- Shutdown proof passed: tracked cancel by cloid was attempted, final open orders were empty, and the independent remote open-orders check returned `final_open_orders_empty=true`.
- T008 ledger found `fill_count=0`, `maker_fill_count=0`, `live_realized_pnl_proof=false`, and `realized_pnl_proof_status=fail_closed_no_realized_live_pnl`.
- Artifact health check found `93` files and `0` empty files in the rerun output directory. Redaction scan found no raw 64-hex secret material; matches were expected boundary/manifest field names and redacted credential-source fields.
- M2 remains blocked on no live maker fill / fee / inventory / realized PnL proof. Do not enter M3, do not claim stable PnL, and do not weaken `Alo`, `<=0.005 BTC`, or T008 fail-closed boundaries.

## 0622T004 Execution Started

- `0622T004` is now `执行中`.
- Execution started on `2026-06-22 16:46 CST`.
- The business thread is implementing the event-driven current-candidate watcher/live path first, then will run focused local verification before any remote refresh/final-gate/live-capable step.
- Live execution remains gated by local tests, git-safe remote refresh, final gate go, current event-driven candidate eligibility, immediate guard pass, and the unchanged `<=2` post-only `Alo` submission cap.

## 0622T004 Prepared Task

- `0622T004` has been created as the next formal M2 repair task and is `待执行`.
- Scope is one event-driven implementation/live-calibration task: convert the `0622T003` same-process watcher path from fixed-window batch candidate generation to rolling current L2/BBO + rolling trade-flow evaluation on each relevant public event.
- The goal is to remove the remaining opportunity half-life inside public collection / candidate selection / immediate guard, not to loosen safety gates.
- Candidate context must be generated from the current in-memory snapshot used by immediate guard; old batch candidates, pulled-back artifact candidates, and quotes no longer at current touch must fail closed.
- Target latency is `candidate_event_to_guard_start <= 500ms`; selected current candidate age must fail closed when `>1.0s`.
- Live boundary remains unchanged: public-only waiting phase, Hyperliquid `Alo` post-only, no taker/crossing, no one-tick-back workaround, no cap relaxation, dynamic size hard cap `<=0.005 BTC`, at most `2` real submissions, tracked cancel, independent final open-orders proof, and T008 ledger fail-closed.
- M2 remains blocked until live maker fill plus fee/inventory/realized PnL evidence passes T008.

## 0622T003 Execution Update

- `0622T003` business execution is complete and is now `待验收`.
- Implementation commits: `1b063a7` (`0622 add same-process M2 watcher live path`) and `5c58af1` (`0622 tighten same-process M2 trigger freshness`).
- The task added a same-process remote watcher/live path so public watcher trigger, selected candidate context, immediate guard, private preflight, post-only `Alo` submit/cancel path, pullback, final open-orders proof, and T008 ledger can run in one `awsserver1` process after final gate go.
- Focused verification passed: Hyperliquid M2 focused tests returned `29 passed`, `py_compile` passed, CLI help checks passed, and `git diff --check` passed.
- Formal final rerun used `--iteration-seconds 3` into `local_live_analysis/hyperliquid_tiny_live_m2_same_process_watcher_0622T003_rerun_short_iter/`, refreshed remote to `5c58af1bdd10317ee1953060a37b37d03431188d`, and final gate returned `tiny_live_ready_for_controller_go` with `allow_create_0617T008=true`.
- Watcher ran `163.946857s`, completed `54` iterations, collected `l2Book=81`, `trades=181`, `subscription_ack=108`, `reconnect_count=0`, expanded `1900` trade events, evaluated `150` candidates, and found `1` eligible candidate.
- Selected candidate was buy `quality_a` at quote `64227`, size `0.005 BTC`, source same-side top qty `0.00033 BTC`, source top order count `2`, top-depth multiple `0.066`, strict-through `0.20921 BTC`, and at-or-through `0.20954 BTC`.
- Same-process boundary held before submit: `public_waiting_phase_private_or_order_endpoint_called=false`, `controller_pullback_before_order=false`, and `separate_live_window_process=false`.
- Immediate guard failed closed before any order submission because candidate age was `3.711s` versus max `3.0s`, selected quote `64227` was no longer current touch, current bid/ask had moved to `64219/64220`, current same-side top qty was `20.52451 BTC`, top order count was `49`, and current top-depth multiple was `4104.902x`.
- Result: `live_submissions_count=0`, `live_window_triggered=false`, `fill_count=0`, `maker_fill_count=0`; independent remote open-orders proof was empty and T008 ledger returned `fail_closed_no_realized_live_pnl`.
- Artifact health check found required final artifacts present with `714` files and `0` empty files. Redaction scan found no credential/private-key/signature values.
- M2 remains blocked. The repair removed the controller pullback / separate-window latency path, but the selected public candidate still decayed before the immediate same-process guard could safely submit.

## 0622T003 Execution Started

- `0622T003` is now `执行中`.
- Execution started on `2026-06-22 CST`.
- The business thread is implementing a same-process watcher-triggered maker-order path first, then will run focused local verification before any remote refresh/final-gate/live-capable step.
- Live execution remains gated by local tests, git-safe remote refresh, final gate go, public watcher eligibility, immediate current-candidate guard, and the unchanged `<=2` post-only `Alo` submission cap.

## 0622T003 Prepared Task

- `0622T003` has been created as the next formal M2 task.
- Scope is intentionally one repair task: merge the public watcher and order path into the same `awsserver1` remote process so a trigger can use the same selected current candidate without controller pullback and without starting a separate long public precheck window.
- The immediate submission path remains guarded: current L2/BBO must still make the selected candidate post-only/non-crossing, quote age must be within a strict threshold, size must remain `<=0.005 BTC`, and TIF must be Hyperliquid `Alo`.
- Live boundary remains unchanged: at most `2` real submissions, no taker/crossing, no one-tick-back workaround, no cap relaxation, tracked cancel, final open-orders proof, independent remote open-orders check, and T008 ledger fail-closed.
- Expected fail-closed states are part of the task: no eligible window, stale candidate before order, final gate failure, no fill, open-orders proof failure, or missing T008 fee/inventory/realized PnL evidence.
- M2 remains blocked until live maker fill plus fee/inventory/realized PnL evidence passes T008.

## 0622T002 QA Update

- `0622T002` QA is `阻塞`.
- QA accepted the implementation and safety boundaries for the time-boxed public watcher: focused regression passed with `25 passed`, `py_compile` and `git diff --check` passed, required artifacts were present and non-empty, artifact scan found `163` files and `0` empty files, and redaction scan found only allowed field names / boolean flags.
- Remote refresh and final gate passed; watcher phase stayed public-only with `private_or_order_endpoint_called=false`, `real_order_endpoint_called=false`, and `real_cancel_endpoint_called=false`.
- Watcher found `1` eligible `quality_a` buy candidate out of `54`, but the separate triggered live window reran current fresh-touch gating after handoff and found `0` allowed candidates, so `live_submissions_count=0`.
- Independent remote open-orders proof returned empty, and T008 ledger returned `live_realized_pnl_proof=false` with `realized_pnl_proof_status=fail_closed_no_realized_live_pnl`.
- M2 remains blocked. The actionable bottleneck is trigger-to-order staleness / decoupling between watcher evidence and live-window submission, not open-orders shutdown or T008 arithmetic.

## 0622T002 Execution Update

- `0622T002` business execution is complete and is now `待验收`.
- Implementation commit: `b1c1ff9` (`0622 add timeboxed M2 public watcher`).
- Formal run refreshed `awsserver1:/home/admin/hftbacktest-cross-exchange` from `ad6080cfe06e39cf04e5b93bfddc418d05b96c17` to `b1c1ff938f559ce99e705ba4130ed83151d68952` with a `19402` byte incremental bundle; remote branch remained `cross-exchange`, dirty count `0`, and Hyperliquid SDK availability `true`.
- Final gate returned `tiny_live_ready_for_controller_go`, `allow_create_0617T008=true`, and `blocking_reasons=[]`.
- Public watcher ran `120.451756s` of the `600s` timebox and stopped after iteration `6` because it found `1` eligible `quality_a` buy candidate out of `54` evaluated candidates. Watcher public counts were `l2Book=28`, `trades=175`, `subscription_ack=12`, `reconnects=0`; diagnosis expanded these into `923` trade events and `54` candidates.
- The selected watcher candidate was buy at `64407`, same-side top qty `0.00016 BTC`, order count `1`, strict-through `0.41541 BTC`, dynamic size `0.005 BTC`, `quality_a`, `fresh_or_reset_supported`.
- The triggered live window correctly reran current fresh-touch gating before order submission. By then the current public precheck had changed; it found `10` fresh-touch candidates but `0` full quality-gate allowed candidates, so `fresh_touch_submitted_count=0`, `real_order_endpoint_called=false`, and `real_cancel_endpoint_called=false`.
- Independent remote open-orders proof returned empty, and T008 ledger returned `live_realized_pnl_proof=false` with `realized_pnl_proof_status=fail_closed_no_realized_live_pnl`.
- M2 remains blocked. This run proves the watcher can find a candidate, but the trigger-to-live handoff is too slow or too decoupled from the exact public window to place an order before the opportunity decays.

## 0622T002 Execution Started

- `0622T002` is now `执行中`.
- Execution started on `2026-06-22 CST`.
- The business thread is adding a bounded public watcher around the accepted `0622T001` fresh-touch gate first, then will run focused local verification before any remote refresh/final-gate/watcher/live step.
- Live execution remains gated by local tests, git-safe remote refresh, final gate go, public watcher eligibility, and the unchanged `<=2` post-only `Alo` submission cap.

## 0622T002 Prepared Task

- `0622T002` has been created as the next formal M2 task.
- Scope is one bounded loop, not another open-ended diagnosis split: implement and execute a time-boxed public-only L2/trades watcher, wait for a `0622T001` fresh-touch quality-gate eligible current-window candidate, and only then trigger one controlled maker-only live micro-window.
- Watcher phase is public-only and must not read credentials, call private/account/order endpoints, submit orders, cancel orders, or start a live bot.
- Live boundary remains unchanged: Hyperliquid `Alo` post-only, no taker/crossing, no one-tick-back workaround, no cap relaxation, dynamic size hard cap `<=0.005 BTC`, at most `2` real submissions, tracked cancel, independent final open-orders proof, and T008 ledger fail-closed.
- Expected fail-closed states are part of the task: no eligible window during the bounded watcher, final gate failure, watcher collection failure, no fill after a live trigger, open-orders proof failure, or missing T008 fee/inventory/realized PnL evidence.
- M2 remains blocked until a live maker fill plus fee/inventory/realized PnL evidence passes T008.

## 0622T001 Execution Update

- `0622T001` business execution is complete and is now `待验收`.
- Implementation commits: `81cb084` (`0622 add fresh-touch M2 live gate`) and `ad6080c` (`0622 fix fresh-touch allowed count`).
- Formal gated loop refreshed remote checkout, reran final gate to go, ran public precheck, and evaluated the new `fresh_touch` session gate.
- Public precheck: `l2Book=4`, `trades=17`, `subscription_ack=2`, `reconnects=0`, `close_reason=duration_elapsed`; diagnosis produced `8` candidates with buy strict-through `1/4`, sell strict-through `0/4`, and public depletion `0/8`.
- Fresh-touch gate result: `fresh_touch_candidate_count=8`, full quality-gate allowed candidates `0`, submitted live orders `0`, `real_order_endpoint_called=false`, and independent remote open-orders check returned empty.
- T008 ledger returned `live_realized_pnl_proof=false` and `realized_pnl_proof_status=fail_closed_no_realized_live_pnl`; M2 remains blocked.
- After the allowed-count artifact/code fix, a no-order remote sync moved `awsserver1` to `ad6080cfe06e39cf04e5b93bfddc418d05b96c17` and final gate still returned `tiny_live_ready_for_controller_go`.

## 0622T001 Execution Started

- `0622T001` is now `执行中`.
- Execution started on `2026-06-22 10:20 CST`.
- The business thread is implementing the accepted `fresh_touch_size_by_throughput_session_gate` policy first, then will run focused local verification before any remote refresh/final-gate/live step.
- Live execution remains gated by local tests, git-safe remote refresh, final gate go, public session eligibility, and the `<=2` post-only `Alo` submission cap.

## 0622T001 Prepared Task

- `0622T001` has been created as the next formal M2 task.
- Scope is intentionally one task, not another diagnostic/design split: implement `fresh_touch_size_by_throughput_session_gate` in `fill_window` / `fill_loop`, run focused local verification, refresh remote checkout, rerun final gate, run public precheck/session-gate, and if eligible execute one controlled live micro-window.
- Live boundary: at most `2` real post-only `Alo` submissions, dynamic size hard cap `<=0.005 BTC`, no one-tick-back workaround, no taker/crossing, no cap relaxation, no default-on, and no promotion.
- Expected fail-closed states are part of the task: no eligible micro-window, no fill, missing fee/inventory/mark evidence, final gate failure, open-orders proof failure, or T008 ledger failure.
- M2 remains blocked until live maker fill plus fee/inventory/realized PnL evidence passes T008.

## 0619T002 Redesign Update

- `0619T002` QA is `已通过`.
- The QA result accepts the design-only redesign contract and confirms it did not change code, place orders, read credentials, refresh remote checkout, rerun final gate, or create live/implementation artifacts.
- The redesign is based on accepted `0618T011` / `0618T012` evidence and the blocked `0619T001` live retry result.
- `0619T001` still allowed submissions at about `120.25x` and `199.18x` same-side top depth multiples; one quote rested `15s` without fill and one lost touch after about `1.01s`. That is still too loose for M2 fill acquisition.
- New policy shape:
  - fresh-touch touch-only entry; no one-tick-back workaround
  - `quality_a`: `<=20x` top-depth multiple, `<=6` top orders, hold `<=3s`
  - `quality_b`: `20x-100x` top-depth multiple, `<=12` top orders, hold `<=1s`
  - dynamic size `min(bucket_cap, 0.25 * recent_same_side_at_or_through_trade_qty_btc_last_3s, 0.005 BTC)`
  - default `buy_only`
  - no fixed time-of-day decision yet; execution remains gated to current precheck-confirmed micro-windows until a later cross-hour scorecard exists
- No code changed, no order was placed, and no remote/final-gate action was taken in this redesign task.
- M2 remains blocked; this accepted redesign is only the next implementation contract.

## 0619T001 QA Update

- `0619T001` QA is `阻塞`.
- The task implemented and executed the flow-aware M2 maker-only retry repair while preserving post-only `Alo`, T008 ledger fail-closed, tracked cancel, final open-orders proof, and same-or-smaller `0.00999 BTC` caps.
- The first loop attempt stopped before final gate/live order because a full `162M` git bundle upload timed out; the refresh path was repaired to use a `60461` byte incremental bundle from the observed remote commit.
- The formal rerun refreshed `awsserver1:/home/admin/hftbacktest-cross-exchange` from `cross-exchange:5391b79439a4f0cb24fd40e4e6aa4b8de53f73e3:0` to `cross-exchange:7d0e813704addc5412e974c79b0eee922075a544:0`.
- Final gate returned `tiny_live_ready_for_controller_go`, `allow_create_0617T008=true`, and `blocking_reasons=[]`.
- Public-flow precheck passed with `l2Book=10`, `trades=19`, `subscription_ack=2`, `reconnects=0`, `close_reason=duration_elapsed`; diagnosis found `6` candidates, `4/6` strict-through, `6/6` touch-trade, `0/6` public top+order depletion, and `4/6` adverse lost-touch.
- The live window completed `6` flow-aware attempts: attempt 1 submitted sell at `62897.0` and reached `resting`; attempt 2 submitted buy at `62880.0` and quote-aging guard canceled/requoted after `lost_touch+adverse_drift`; attempts 3-6 were skipped as `skip_crowded_touch`.
- Window shutdown passed with `real_order_endpoint_called=true`, `real_cancel_endpoint_called=true`, `post_only_tif=Alo`, `crossing_guard_status=pass`, `final_open_orders_count=0`, and independent remote open-orders check `final_open_orders_empty=true`.
- No fill occurred: `fill_count=0`, `maker_fill_count=0`, aggregate live fill ledger is empty.
- T008 ledger returned `live_realized_pnl_proof=false` and `realized_pnl_proof_status=fail_closed_no_realized_live_pnl`.
- M2 remains blocked; M3 must not start.

## 0619T001 Prepared Task

- `0619T001` has been created as the next formal M2 task: maker-only flow-aware retry repair for quote aging and side selection.
- The task converts `0618T012` public-flow conclusions into execution behavior: no blind buy/sell alternation, shorter/stale-aware quote holding, lost-touch/adverse-drift cancellation, side-specific public-flow scoring, and crowded-touch queue filtering.
- It preserves Hyperliquid `Alo` post-only, T008 ledger fail-closed, tracked cancel, final open-orders proof, and the same or smaller caps.
- It may submit real orders only after local self-tests pass, `awsserver1` remote checkout refresh succeeds, final gate returns go, and flow-aware guards produce an allowed candidate.
- If gate/precheck/flow guards block, if no fill occurs, or if T008 ledger cannot reconcile fill/fee/inventory/PnL evidence, the task must end `阻塞` and M2 remains incomplete.

## 0618T012 Public Flow Diagnosis Update

- `0618T012` QA is `已通过`.
- The task implemented a read-only Hyperliquid public L2/trades flow diagnosis and preserved no-live/no-order/no-credential/no-private-endpoint boundaries.
- Local direct public collection from this machine failed with repeated Hyperliquid public API/WebSocket SSL EOF before any subscription messages, so that empty local attempt was recorded only as a collection-path blocker.
- A public-only `awsserver1` collection succeeded without git refresh, credential reads, private/account/order endpoints, live orders, or strategy process: `l2Book=56`, `trades=122`, `subscription_ack=2`, `reconnects=0`, `close_reason=duration_elapsed`.
- The pulled-back raw sample was analyzed locally into `56` book events, `356` individual trade events, and `38` passive touch-quote candidates using `0.00999 BTC`, `45s` hold, and `5s` candidate stride.
- Hypothesis status:
  - `queue_too_deep=supported`: public top+order depletion proxy reached only `7/38` candidates.
  - `no_trade_through=rejected_for_sample`: strict trade-through appeared in `21/38` candidates and touch trades in `34/38`.
  - `wrong_time_of_day=inconclusive`: one `117.948s` sample in UTC hour `9` is insufficient for time-of-day selection.
  - `wrong_side=supported`: buy had `6/19` public-depletion candidates and `13/19` strict-through candidates, while sell had `1/19` and `8/19`.
  - `quote_aging_or_fast_drift=supported`: adverse lost-touch occurred in `21/38` candidates.
- Interpretation: the sampled no-fill blocker is not absence of trade-through. The stronger public-flow blockers are crowded touch queues, quote aging / fast BBO drift, and side asymmetry, with sell materially worse than buy in this sample.
- M2 remains blocked because this is public-flow proxy evidence only: no exact queue priority, no private order lifecycle, no live maker fill, no fee/inventory proof, and no realized PnL proof.
- Next M2 work should not be another blind same-caps retry. If continuing toward a retry, first design a maker-only repair that addresses quote aging/fast drift and side selection while preserving `Alo`, T008 ledger fail-closed, and same or smaller caps.

## 0618T011 No-Fill Diagnosis Update

- `0618T011` QA is `已通过`.
- The task was no-network/no-live and consumed only local `0618T009` / `0618T010` pulled-back artifacts.
- It analyzed `9` maker-only `Alo` attempts: `6` buy / `3` sell, `9/9` no-fill, `9/9` same-side touch join, median spread proxy `1` tick.
- Defensible same-side public depth proxy exists for `4/9` attempts only: the three T009 windows and T010 attempt 1. Those top-depth multiples ranged from `21.83x` to `1921.75x` of the `0.00999 BTC` order, and are explicitly public-depth proxy only, not exact queue priority.
- T010 attempts 2-6 have attempt BBO but no per-attempt full depth, so they are marked `per_attempt_depth_missing`.
- Evidence gaps remain: exact queue position, trade-through at quote, per-attempt post-L2, side/time-of-day coverage, and realized PnL.
- Design decision: do not continue blind same-caps live retry. Next M2 work should be a read-only public L2/trades flow diagnosis before any later maker-only retry.
- M2 remains blocked on live maker fills; M3 must not start.

## 0618T010 Adaptive Fill Retry Update

- `0618T010` QA is `阻塞`.
- The fill-acquisition repair added adaptive maker-only cancel/requote under the same caps.
- Live retry used one 10-minute-bounded window with `requote_attempts=6`, `quote_hold_seconds=45`, `side_policy=alternate`, and `quote_offset_ticks=0`.
- Remote `/home/admin/hftbacktest-cross-exchange` ended at `cross-exchange:5391b79439a4f0cb24fd40e4e6aa4b8de53f73e3:0`.
- Final gate before live retry returned `tiny_live_ready_for_controller_go`, `allow_create_0617T008=true`, and no blockers.
- Attempt results: `6/6` attempts reached `resting`, alternated buy/sell, used post-only `Alo`, and passed crossing guard.
- Shutdown result: tracked cancel path executed, final open orders were 0, and independent remote open-orders check also returned 0.
- Fill/PnL result: `fill_count=0`, `maker_fill_count=0`, aggregate fill ledger empty, and T008 ledger returned `fail_closed_no_realized_live_pnl`.
- M2 remains blocked. Continuing with blind same-caps retries has low information value; next step should diagnose no-fill causes before another live retry.

## 0618T009 M2B Fill Loop Update

- `0618T009` QA is `阻塞`.
- The M2B loop safely executed under the approved tiny-live envelope but did not complete M2.
- Remote git-safe refresh used bundle + `git merge --ff-only`; remote `/home/admin/hftbacktest-cross-exchange` ended at `cross-exchange:7280fbd0f0c6f4e684cad96f2775bc2bae7ca70a:0`.
- Final gate immediately before live windows returned `tiny_live_ready_for_controller_go`, `allow_create_0617T008=true`, and no blockers.
- Window results: `3/3` reached `resting`, used post-only `Alo`, called real private/order/cancel endpoints, used tracked cancel, recorded `shutdown_proof_status=pass`, and ended with `final_open_orders_count=0`.
- Fill/PnL result: all three windows had `fill_count=0`; aggregate `aggregate_live_fill_ledger.csv` is empty.
- T008 ledger rerun in `live_pulled_back` mode returned `live_realized_pnl_proof=false` and `realized_pnl_proof_status=fail_closed_no_realized_live_pnl`.
- Independent remote open-orders check returned `final_open_orders_count=0`.
- M2 remains blocked on live maker fills; do not enter M3 or claim stable PnL.

## 0618T008 M2A Ledger Update

- `0618T008` QA is `已通过`.
- M2A implemented a no-network Hyperliquid PnL ledger/reconciler for fills, fees/rebates, inventory delta, slippage, replay/live proof limits, source completeness, and overclaim gates.
- Official artifacts: `local_live_analysis/hyperliquid_tiny_live_m2_pnl_ledger_0618T008/`.
- Against accepted M1 artifacts, the ledger found `windows_found=3` but `live_realized_pnl_proof=false` and `realized_pnl_proof_status=fail_closed_no_realized_live_pnl`; M1 canary evidence is not PnL proof.
- Fixture artifacts under `local_live_analysis/hyperliquid_tiny_live_m2_pnl_ledger_0618T008_fixture/` prove arithmetic only: `gross_pnl_usdc=0.26`, `fee_usdc=0.125268`, `net_pnl_usdc=0.134732`, `inventory_delta_btc=0.01`, `slippage_usdc=0.0`.
- No order was placed, no credential was read, no private/account/order endpoint was called, and no live bot was started.
- Next main task is `0618T009` M2B: controlled tiny-live fill loop plus PnL reconciliation using the T008 ledger.

## 0618T007 M1 Canary Loop Update

- `0618T007` QA is `已通过`.
- M1 completed with one formal task and three independent Hyperliquid tiny-live canary windows under the approved strict caps.
- Git-safe remote refresh used bundle + `git merge --ff-only`; remote `/home/admin/hftbacktest-cross-exchange` ended at `cross-exchange:a2e550214ecca865f075e670c5cef0a043cdb049:0`.
- Final gate immediately before M1 returned `tiny_live_ready_for_controller_go`, `allow_create_0617T008=true`, and no blockers.
- Window results: `3/3` reached order status `resting`, called private/order/cancel endpoints, used tracked cancel, recorded `shutdown_proof_status=pass`, and ended with `final_open_orders_count=0`.
- `schedule_cancel_endpoint_called=false` for all three windows; M1 did not rely on the account-ineligible scheduled-cancel/dead-man switch.
- M1 does not prove realized PnL, fees/rebates, slippage, inventory accounting, stable PnL, maker viability, default-on, promotion, or scale-up. Next main milestone is M2 real PnL and cost accounting.

## 0618T006 Gate Refresh Update

- `0618T006` QA is `已通过`.
- Remote checkout `/home/admin/hftbacktest-cross-exchange` is now confirmed synchronized with local HEAD `d37438e0c`.
- Remote facts: branch `cross-exchange`, dirty count `0`, Python `/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python`, Python version `Python 3.13.5`.
- Final gate rerun now returns `tiny_live_ready_for_controller_go` with `allow_create_0617T008=true` and `blocking_reasons=[]`.
- This is only the M1 precondition refresh; it does not execute live/canary orders and does not prove PnL or maker viability.

## 0618T005 M0 Baseline Update

- `0618T005` QA is `已通过`.
- M0 used the minimum-task path: one read-only business verification task plus QA.
- No order was placed, no credential value was read, no private/account/order endpoint was called, and no live bot was started.
- Read-only signal / quote replay rerun preserved the primary candidate: `75` ticks / persistence `2`, `654` theoretical intents, `313` buy / `341` sell, `2.4269%` intent rate, and `8/8` samples with any intent.
- Read-only optimistic proxy rerun preserved the official `canonical_7` result at `75` ticks / persistence `2` / `1000ms`: `295.985 USDC`, mean `40.932789` ticks per intent, `7/7` positive samples under the explicit optimistic upper-bound assumption.
- Accepted `0618T004` canary artifacts still prove the historical interface path: order reached `resting`, tracked cancel succeeded, final `open_orders=[]`, shutdown proof passed, and redaction flags remained false for credentials / secret values / raw signatures.
- The M0 final gate rerun is intentionally read-only and currently fails closed with `allow_create_0617T008=false`, blocker `remote_execution_checkout_not_synced_or_invalid`; saved remote state is `cross-exchange:52b5b9541:0`, while current local gate commit is `d37438e`.
- M0 is complete as evidence-chain baseline verification, but it is not live authorization. Before M1, the remote execution checkout must be refreshed/synced and the final gate rerun must pass.

## 0618T001 QA Update

- `0618T001` QA is `已通过`.
- Executor repair, no-order self-test artifacts, remote pullback, and final gate fail-closed behavior were accepted.
- `0618T002` is now unblocked at the QA prerequisite level and may execute next.

## 0618T002 QA Update

- `0618T002` QA is `已通过`.
- The official `hyperliquid-python-sdk==0.24.0` is available locally via `/home/molly/anaconda3/bin/python` and on `awsserver1` via `/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python`.
- Local and remote SDK readiness checks passed without wallet-backed client construction, credentials, private/account/order endpoints, signing, nonce, websocket, real order, or cancellation behavior.
- The `0618T002` final gate output is `tiny_live_ready_for_controller_go` with `allow_create_0617T008=true`.
- This does not execute or create `0617T008`; any live/canary order task still needs separate task creation and execution.

## 0618T003 QA Update

- `0618T003` QA is `已通过`.
- The Hyperliquid credential-shaped fields are located in `/home/admin/XEMM_rust/.env` and `/home/admin/XEMM_rust_latest/.env` as `HL_WALLET` and `HL_PRIVATE_KEY`.
- The two `config.json` files contain Hyperliquid configuration keys but did not provide the private-key-shaped credential hit.
- No token/private-key values were returned or written to repo artifacts; only paths and key names were recorded.

## 0618T004 Prepared Task

- `0618T004` has been created as the next formal task: Hyperliquid tiny-live real-order canary interface validation.
- Scope: verify the authenticated private-read, order, cancel, schedule-cancel, and shutdown interface chain needed for later livetest using the credential location results from `0618T003`.
- It must remain a minimal canary with strict caps, immediate cancel / shutdown evidence, redacted artifacts, and pullback for QA.
- It is not a continuous live strategy task and must not relax `10min / 0.01 BTC / post-only / max loss cap`.

## 0618T004 Execution Update

- `0618T004` business execution is complete and is `待验收`.
- `awsserver1:/home/admin/hftbacktest-cross-exchange` was synced to `63f176154`.
- The real-order canary used the credential source path `/home/admin/XEMM_rust_latest/.env` without returning or writing secret values.
- The authenticated interface chain was exercised: private read, real `Exchange.order`, `Exchange.cancel`, `Exchange.cancel_by_cloid`, `Exchange.schedule_cancel`, and final `Info.open_orders`.
- The canary order response was `resting`; tracked cancel succeeded; final open orders were empty; shutdown proof is `pass`.
- `schedule_cancel` was reachable but rejected by Hyperliquid account eligibility because the account has not met the required traded-volume threshold. Later live tasks must not rely on dead-man switch unless this changes.
- Final canary recommendation is `hyperliquid_tiny_live_real_order_canary_ready_for_qa`.
- T004 final gate output is `tiny_live_ready_for_controller_go` with `allow_create_0617T008=true`.

## 0618T004 QA Update

- `0618T004` QA is `已通过`.
- QA accepted the real-order canary evidence: the post-only `Alo` BTC order reached `resting`, tracked cancel succeeded, final `open_orders` was empty, and redacted artifacts preserved credential/order identifier boundaries.
- QA refreshed the remote final-gate input after the report commit sync; current remote state is `cross-exchange:52b5b9541:0`.
- T004 final gate remains `tiny_live_ready_for_controller_go` with `allow_create_0617T008=true`.
- Residual limitation: `schedule_cancel` / dead-man switch is currently unavailable for this account due traded-volume eligibility, so later live tasks must use tracked cancel plus final open-orders proof unless this is separately revalidated.

## 0616T007 QA Update

- `0616T007` was created and executed as the `awsserver1` live-capable preflight dry-run after `0616T006` QA passed.
- QA status is `阻塞`.
- SSH to `awsserver1` succeeded and remote dry-run artifacts were created under `/home/admin/hftbacktest_live_artifacts/0616T007_preflight`, then pulled back locally to `local_live_analysis/hyperliquid_awsserver1_preflight_dry_run_0616T007/`.
- Blocking facts: remote repo path is `/home/admin/hft_live/hftbacktest`, branch is `master` instead of required `cross-exchange`, dirty status count is `29`, remote `conda` is unavailable, and remote `rsync` is unavailable.
- Boundary facts: no credential read, no private endpoint, no account query, no order placement, no cancellation, no amendment, and no live bot startup occurred.
- The auto loop must stop here. `0616T008` must not be created or executed until a later preflight dry-run on `awsserver1` passes QA.
- Follow-up controller clarification: `/home/admin/hft_live/hftbacktest` on `master` is the Binance maker execution route and should remain separate. A new cross-exchange remote path such as `/home/admin/hftbacktest-cross-exchange` may be created for this branch. Remote preflight may use system `python3` if recorded explicitly; conda is not required on `awsserver1` for this path.

## 0617T001 QA Update

- `0617T001` created a separate `awsserver1` checkout at `/home/admin/hftbacktest-cross-exchange` for the current local `cross-exchange` branch.
- QA status is `已通过`.
- The existing Binance maker route `/home/admin/hft_live/hftbacktest` was preserved as `master:703c149:29`.
- The new cross-exchange route is `cross-exchange:7642b16:0`.
- The selected remote Python for this path is `/usr/bin/python3`, version `Python 3.13.5`; remote conda is not required for this path.
- Artifacts were pulled back with `scp` to `local_live_analysis/hyperliquid_awsserver1_cross_exchange_python3_preflight_0617T001/` and checksums validated.
- No credential read, private endpoint, account query, order placement, cancellation, amendment, or live bot startup occurred.
- Next action: create a new live-capable preflight dry-run over `/home/admin/hftbacktest-cross-exchange`; do not jump directly to `0616T008`.

## 0617T002 QA Update

- `0617T002` repeated the `awsserver1` cross-exchange python3 preflight on `/home/admin/hftbacktest-cross-exchange`.
- QA status is `已通过`.
- The repeated remote state is `cross-exchange:7642b16:0`.
- The selected remote Python remains `/usr/bin/python3`, version `Python 3.13.5`.
- Artifacts were pulled back with `scp` to `local_live_analysis/hyperliquid_awsserver1_cross_exchange_python3_preflight_0617T002/` and checksums validated.
- No credential read, private endpoint, account query, order placement, cancellation, amendment, or live bot startup occurred.
- Next action: create the final live-capable preflight/operator task that binds `/home/admin/hftbacktest-cross-exchange`, `/usr/bin/python3`, `scp` pullback, and the approved `0616T008` caps. Do not execute real orders before that task passes QA.

## 0617T003 Prepared Task

- `0617T003` has been created as the final live-capable preflight/operator task before any `0616T008` live execution may be created.
- Scope: bind `/home/admin/hftbacktest-cross-exchange`, `/usr/bin/python3`, `scp` pullback, and the approved `0616T008` caps into final operator/preflight artifacts.
- It must preserve the existing Binance maker route `/home/admin/hft_live/hftbacktest` and must not place/cancel/amend orders, query accounts, call private endpoints, read credentials, start a live bot, deploy, promote, prove PnL, or claim maker viability.
- Business execution and QA are `已通过`.
- Official artifacts: `local_live_analysis/hyperliquid_tiny_live_final_live_capable_preflight_0617T003/`.
- Final recommendation: `hyperliquid_tiny_live_final_live_capable_preflight_ready_for_qa`.
- `approved_caps.csv` preserves the `0616T008` caps and marks `real_orders_allowed` as allowed only inside a separately dispatched `0616T008`; `0617T003` itself did not execute orders.
- Next step, if requested by total control, is to create the separate `0616T008` live execution task.

## 0617T004 Prepared Task

- `0617T004` has been created as the required signal / quote policy protocol before any `0616T008` live execution.
- Scope: define the Binance-lead / Hyperliquid-lag maker signal, side mapping, quote placement rule, size/cap policy, cancel/stop policy, and audit fields.
- It must either source conservative thresholds from accepted local artifacts or fail closed with `hyperliquid_tiny_live_signal_quote_policy_needs_threshold_calibration`.
- It must not place/cancel/amend orders, query accounts, call private endpoints, read credentials, start a live bot, deploy, promote, prove PnL, or claim maker viability.

## 0617T005 Replay Task

- `0617T005` was dispatched after `0617T004` QA passed and QA is now `已通过`.
- Scope: read-only signal / quote replay over the existing accepted cross-exchange public/read-only datasets, using the final `0617T004` protocol as the rule source.
- Intended replay outputs: `basis_mid` / `basis_mid_ticks`, threshold and persistence state, Hyperliquid maker buy/sell intent mapping, theoretical quote price and quote distance, post-only/crossing checks, cancel/reject reasons, trigger counts, side distribution, stale/data-gap rejection counts, and simulated cap-trigger diagnostics for `0.01 BTC` order size and `0.04 BTC` max position.
- It must not place/cancel/amend orders, query accounts, call private endpoints, read credentials, start a live bot, deploy, promote, prove PnL, claim real fills, claim real inventory, claim post-only reject behavior, claim queue priority, or claim maker viability.

## 0617T004 Execution Update

- `0617T004` business execution is complete and has been written to `待验收`.
- The protocol defines `basis_mid = binance_mid - hyperliquid_mid`, `basis_mid_ticks = basis_mid / hyperliquid_tick_size`, positive-signal buy intent, negative-signal sell intent, maker-only/post-only quoting, cancel/stop rules, and required runtime audit fields.
- Live threshold status is `blocked_for_live_execution`; accepted artifacts support directional structure but not a defensible absolute live cutoff.
- Next step: separate read-only signal / quote replay and threshold calibration before any `0616T008` live execution.
- QA status is `已通过`.

## 0617T005 Execution Update

- `0617T005` was formally dispatched after `0617T004` QA passed.
- Business execution is complete and QA is `已通过`.
- Full quote replay now resolves all locally present pricing-signal inputs: `0601T005` plus the `7` historical event-mode `pricing_signal_rows.csv` files referenced by `0609T008`.
- Replay consumed `8` pricing-signal files, `161455` raw pricing rows, and `26948` de-duplicated decision rows across threshold grid `10,20,30,40,50,75,100` ticks and persistence grid `1,2,3`.
- Source availability now confirms all seven historical manifest samples are `local_direct_file_available=true` and `replay_source_used=pricing_signal_rows`; the earlier missing-file conclusion was a local path-resolution issue.
- Calibration candidate: `75` ticks with `2` observations of persistence, `654` theoretical intents (`313` buy / `341` sell), `2.4269%` intent rate, and `8/8` sample coverage. Stricter fallback: `75` ticks with `3` observations, `327` intents and `1.2134%` intent rate.
- Final recommendation is `hyperliquid_tiny_live_signal_quote_replay_ready_for_qa`. This read-only result still does not authorize `0616T008` live execution before `0617T006` QA/controller ratification.

## 0617T005 QA Update

- `0617T005` QA status is `已通过`.
- Focused test, replay runner, JSON manifest validation, required artifact non-empty checks, boundary checks, and `git diff --check` passed.
- `docs/qa-acceptance-report.md` now records `0617T005` as the latest effective QA result.
- Next action: execute `0617T006` read-only optimistic PnL proxy using the accepted `0617T004` / `0617T005` rule sources.

## 0617T006 Prepared Task

- `0617T006` has been created as the next formal task: Hyperliquid tiny-live read-only optimistic PnL proxy.
- Scope: estimate a theoretical public-data upper bound over existing accepted pricing-signal rows, assuming theoretical maker intents fill at quote and using future Hyperliquid mid labels for fixed-horizon and oracle-best-horizon settlement.
- It must reconcile the user's "6 datasets" wording against the actual local accepted manifests before computing results; if the exact six-sample set is ambiguous, it must report that explicitly and separate any `canonical_7` / `0617T005_8_input` diagnostic estimate.
- It may use the `0617T004` side mapping and the `0617T005` threshold candidates, especially `75` ticks with persistence `2` and fallback persistence `3`.
- It must not model or claim fill probability, queue priority, private/order lifecycle, post-only reject probability, account inventory, fees/rebates, spread-capture settlement, real PnL, maker viability, deployment readiness, or live authorization.

## 0617T006 QA Update

- `0617T006` QA is `已通过`.
- The user selected `canonical_7` as the formal sample-set口径, superseding the earlier exact `6` dataset wording.
- Official `canonical_7` estimate was computed from `139914` pricing rows and `23353` decision rows.
- Diagnostic `0617T005_8_input` estimate was computed from `161455` pricing rows and `26948` decision rows.
- At `75` ticks / persistence `2` / `1000ms`, `canonical_7` reports `295.985 USDC` optimistic proxy, mean `40.932789` ticks per intent, and `7/7` positive samples.
- At `75` ticks / persistence `2` / `1000ms`, `0617T005_8_input` reports `299.38 USDC` optimistic proxy, mean `39.407661` ticks per intent, and `8/8` positive samples.
- Oracle-best-horizon output is labeled `non_tradeable_oracle_upper_bound` and must not be interpreted as tradable strategy evidence.
- Final recommendation is `hyperliquid_tiny_live_optimistic_pnl_proxy_ready_for_qa`.
- Latest QA report: `.workflow/reports/0617T006-qa.md`; latest fixed QA acceptance document now records `0617T006`.

## 0617T007 Execution Update

- `0617T007` was created and business-executed as the final read-only go/no-go gate before any possible `0617T008` tiny-live execution task.
- It recovered the accepted `0617T006` `canonical_7`口径, prior operator packet/caps, latest controller instruction, and actual `awsserver1` remote facts for `/home/admin/hftbacktest-cross-exchange`.
- The gate final recommendation is `tiny_live_needs_missing_precondition`.
- `allow_create_0617T008=false`.
- Blocking reasons are `remote_execution_checkout_not_synced_or_invalid` and `hyperliquid_real_order_executor_missing_or_unproven`.
- Remote state at gate runtime was `/home/admin/hftbacktest-cross-exchange`, branch `cross-exchange`, commit `7642b16`, dirty count `0`, Python `/usr/bin/python3`, version `Python 3.13.5`; local accepted commit at runtime was `1556a85`.
- No credentials were read, no private endpoint/account query occurred, no orders were placed/cancelled/amended, and no live bot was started.
- `0617T008` has not been created; the auto loop must stop here unless QA accepts T007 and a later repaired gate returns `allow_create_0617T008=true`.

## 0617T007 QA Update

- `0617T007` QA is `已通过` for the final gate quality, but the gate decision is no-go.
- QA reran the focused test, final gate generator, JSON validation, artifact non-empty checks, boundary checks, and `git diff --check`.
- QA rerun manifest records `final_recommendation=tiny_live_needs_missing_precondition` and `allow_create_0617T008=false`.
- Blocking facts remain: remote `/home/admin/hftbacktest-cross-exchange` is `cross-exchange:7642b16:0` while the local T007 gate commit under QA rerun was `93b2178`, and no QA-accepted Hyperliquid real-order executor/post-only/cancel-all/private-order source path exists.
- Latest QA report: `.workflow/reports/0617T007-qa.md`; latest fixed QA acceptance document now records `0617T007`.
- Auto loop step 3 is not executed: `0617T008` was not created and no live order task was run.

## 0618T001 Prepared Task

- `0618T001` has been created as the next formal repair task: Hyperliquid tiny-live minimal real-order executor and final gate repair.
- Scope: implement and QA a minimal live-capable Hyperliquid real-order executor using the accepted official-doc evidence chain, strict caps, post-only `Alo`, max-loss fail-closed logic, cancel-all shutdown proof, remote `awsserver1` preflight/pullback, and an updated final go/no-go gate.
- Required caps remain fixed: `10 minutes`, `0.01 BTC` max order size, `700 USDC` max order notional, `0.04 BTC` max position, `2800 USDC` max position notional, `3000 USDC` max notional, `30 USDC` max loss, `BTC`, post-only only.
- This task must not place real orders, create or execute `0617T008`, run a 10-minute live test, disclose credentials, relax caps, deploy, promote, prove PnL, or claim maker viability.
- It must rerun final go/no-go gate into `local_live_analysis/hyperliquid_tiny_live_final_go_no_go_gate_0618T001/`; only a later separate live task may consume a passing gate.
- Business execution is complete and awaits QA.
- Implemented `examples/hyperliquid/hyperliquid_tiny_live_real_order_executor.py` plus focused tests and documentation.
- Local and `awsserver1` self-tests generated no-order artifacts. Both environments currently record `hyperliquid_sdk_available=false`.
- Repaired final gate consumes the executor manifest, remote state, dependency matrix, accepted caps, and accepted `canonical_7` evidence.
- Final gate result is `tiny_live_needs_missing_precondition`, `allow_create_0617T008=false`, blocker `hyperliquid_official_sdk_dependency_unavailable`.
- No real order, cancel, private endpoint, account query, credential read, live bot, or `0617T008` creation occurred.

## 0618T002 Prepared Task

- `0618T002` has been created as the next dependency-readiness repair task.
- It must not execute until `0618T001` QA is `已通过`.
- Scope: make the official Hyperliquid Python SDK available in the local validation environment and on `awsserver1`, prove no-order SDK surface readiness, rerun executor self-test under `0618T002` artifacts, and rerun final go/no-go gate.
- It remains no-order/no-private/no-account/no-credential and must not create `0617T008`.
- Success can only remove the SDK dependency blocker; real submit-order API success remains for a later separately approved canary/tiny-live task.

## 0616T008 Live Approval

- On `2026-06-17`, the controller approved one limited `0616T008` Hyperliquid tiny-live small-notional execution window, conditional on `0616T006` QA passing and `0616T007` awsserver1 preflight dry-run QA passing first.
- Approved parameters: `symbol=BTC`, `max_order_size=0.01 BTC`, `max_order_notional=700 USDC`, `max_position=0.04 BTC`, `max_position_notional=2800 USDC`, `max_notional=3000 USDC`, `max_loss=30 USDC`, `duration=10 minutes`, `host_machine=awsserver1`, `account_scope=Hyperliquid account configured on awsserver1`, `maker_only/post_only=true`, `real_orders_allowed=true`.
- BTC/USD reference at approval time was `65794.035`, used to set conservative notional caps.
- The approval is limited to `0616T008`; it does not authorize capital scaling, strategy default-on behavior, deployment, promotion, relaxed caps, or later live windows.
- The auto loop should run `0616T006 QA -> 0616T007 dry-run -> 0616T008 tiny-live`, stopping after `0616T008` for QA and post-live evidence analysis.

## 0616T006 Prepared Task

- `0616T006` has been created and dispatched as the next formal task after `0616T005` QA passed.
- Scope: Hyperliquid tiny-live live-capable preflight / operator packet for future execution on `awsserver1` with artifacts pulled back to local validation.
- It may define host preflight requirements, live-capable config schema, approval field handling, inert command/operator packet structure, artifact pullback/checksum policy, local validators, and dry-run artifacts.
- It must keep unapproved fields as `pending_controller_approval` and must not place/cancel/amend orders, start a live bot, query accounts, disclose credentials, implement signing/nonce/user stream, deploy, promote, prove PnL, or claim maker viability.
- Required first outputs are `docs/hyperliquid_tiny_live_live_capable_preflight_operator_packet.md`, `local_live_analysis/hyperliquid_tiny_live_live_capable_preflight_operator_packet_0616T006/**`, `.workflow/reports/0616T006-business.md`, and any focused `examples/hyperliquid/**` validator or tests needed by the business thread.
- Final recommendation taxonomy is `hyperliquid_tiny_live_live_capable_preflight_operator_packet_ready_for_qa` / `hyperliquid_tiny_live_live_capable_preflight_operator_packet_needs_revision` / `hyperliquid_tiny_live_live_capable_preflight_operator_packet_blocked`.
- Business execution is complete and awaiting QA.
- It implemented `examples/hyperliquid/hyperliquid_tiny_live_operator_packet.py`, focused tests, operator packet documentation, local artifacts, and a business report.
- Official artifacts: `local_live_analysis/hyperliquid_tiny_live_live_capable_preflight_operator_packet_0616T006/`.
- Final recommendation: `hyperliquid_tiny_live_live_capable_preflight_operator_packet_ready_for_qa`.
- `live_authorized=false`, `run_window_authorized=false`, and `real_orders_allowed=pending_controller_approval`.

## 0616 Cross-Exchange Auto Loop

- `docs/cross_exchange_auto_loop_protocol.md` has been created as the controller runbook for the current `cross-exchange` auto loop.
- `0616T001` QA is `已通过`: branch correction is complete; the Binance `BTCUSDT` `0615T009` live path is stopped for this branch.
- `0616T002` QA is `已通过`: Hyperliquid private/order readiness boundary is defined as design-only.
- `0616T003` QA is `已通过`: Hyperliquid no-trading local private order artifact fixture / validator is implemented and verified under `.conda-envs/hft-py38`.
- `0616T004` QA is `已通过`: Hyperliquid cancel-all / shutdown dry-run proof gate is implemented as local fake proof only; exchange-side no-open-order proof remains unproven.
- `0616T005` QA is `已通过`: Hyperliquid tiny-live protocol design is complete and explicitly stops at `stop_for_controller_approval`.
- Auto loop is now stopped by design. No real live task is authorized until the controller approves symbol, max notional, max order size, max position, max loss, duration, host/machine, account scope, and whether real orders are allowed.

## 0616T001 Cross-Exchange Correction

- `0616T001` has been created, dispatched, and executed to `待验收`.
- Current branch is `cross-exchange`; the clarified branch objective is Binance lead / Hyperliquid lag maker strategy research.
- The planned `0615T009` Binance `BTCUSDT` small-cap live path is stopped for this branch and must not be dispatched as the next task.
- `0615T008` is retained only as historical Binance live-risk design reference; it is not the cross-exchange live predecessor.
- `0615T001-T007` are not discarded wholesale: `0615T001`, `0615T006`, and `0615T007` are useful templates; `0615T002-T005` require Hyperliquid-specific migration; none are sufficient to authorize Hyperliquid private/order live execution.
- Official artifacts: `local_live_analysis/cross_exchange_branch_correction_0616T001/`.
- Final recommendation: `cross_exchange_branch_correction_ready_for_qa`.
- Corrected next direction: create a Hyperliquid maker private/order execution-readiness boundary task using `0601T004` / `0601T005` / `0609T002` as the relevant cross-exchange evidence chain.
- Until separate Hyperliquid private/order readiness, cancel-all/shutdown proof, account/inventory/economics source handling, source-chain gate, proof-limited runner, and Hyperliquid-specific live-risk protocol pass QA, no Hyperliquid live order task is authorized.

## Post-0615T005 Forward Plan

- `0615T005` QA is `已通过`; it completes the no-trading local economics fee/rebate read-only source implementation.
- Superseded for the `cross-exchange` branch by `0616T001`: the `0615T009` Binance small-cap live sequence below is historical context only and must not be used as the current next task.
- Total control updated the forward design to support a small-cap live test only as the fourth task in a gated sequence, not as an immediate next action.
- Planned sequence:
  1. `0615T006` source-chain runner-consumption gate / synthesis design.
  2. `0615T007` proof-limited read-only execution evidence runner v1 implementation.
  3. `0615T008` small-cap live-test protocol / risk gate design and dry-run acceptance.
  4. `0615T009` small-cap live test and real-environment data collection.
  5. `0615T010` post-live evidence analysis and decision gate.
- `0615T006` is the next task to create. It should reconcile private order response, account inventory, economics fee/rebate, and replay lifecycle artifacts into a runner-consumption contract.
- `0615T009` is the earliest task that may open a real small-cap live test, and only if `0615T006` / `0615T007` / `0615T008` all pass QA and total control explicitly approves the live window.
- The fourth-task live test must be capped by explicit symbol/config, max notional, max order size, max position, max loss, duration, maker-only/post-only behavior, kill-switch rules, cancel-all/shutdown proof, and required artifact capture.
- `0615T010` must analyze the real-environment data before any decision to repeat, repair, adjust, or prepare another experiment.
- No current task authorizes immediate live execution, strategy default-on behavior, deployment/promotion, PnL proof, capital scaling, or maker viability proof.

## 0615T006 QA Update

- `0615T006` has been created, business execution is complete, and QA is `已通过`.
- Scope: source-chain runner-consumption gate / synthesis design over accepted `0615T003` private order response, `0611T003` replay lifecycle, `0615T004` account inventory, and `0615T005` economics fee/rebate artifacts.
- It produced a design document, source-chain dependency matrix, timestamp reconciliation policy, identity/redaction reconciliation policy, runner input contract, proof-limit taxonomy, fail-closed gate matrix, future runner QA gates, next-task sequence, manifest, boundary validation, and business report.
- Final recommendation: `source_chain_runner_consumption_gate_ready_for_qa`.
- Official artifacts: `local_live_analysis/basis_positive_source_chain_runner_consumption_gate_0615T006/`.
- QA report: `.workflow/reports/0615T006-qa.md`; latest QA acceptance document now records `0615T006`.
- `0615T006` does not authorize runner implementation, endpoint/source collector implementation, credentials/signing/nonce/user stream, real private/order/account/live/economics data reads, order placement/cancellation/amendment, strategy/live/default-on/tiny-live behavior, real metrics, PnL proof, deployment, promotion, or maker viability proof.
- Next auto-loop action: create and execute `0615T007` proof-limited read-only execution evidence runner v1 implementation.

## 0615T007 QA Update

- `0615T007` has been created, business execution is complete, and QA is `已通过`.
- Scope: proof-limited read-only execution evidence runner v1 implementation over accepted local artifacts and the `0615T006` source-chain contract.
- It implemented `examples/binance_tick_mm/execution_evidence_read_only_runner.py`, focused tests, design note, local artifacts, and business report.
- Official artifacts: `local_live_analysis/basis_positive_execution_evidence_read_only_runner_0615T007/`.
- Final recommendation: `proof_limited_read_only_runner_ready_for_qa`.
- QA report: `.workflow/reports/0615T007-qa.md`; latest QA acceptance document now records `0615T007`.
- The runner emitted `10` proof-limited rows, validated missing-source fail-closed behavior, and rejected overclaim requests such as PnL/promotion.
- `0615T007` does not authorize endpoint/source collector implementation, credentials/signing/nonce/user stream, real private/order/account/live/economics data reads, order placement/cancellation/amendment, strategy/live/default-on/tiny-live behavior, real execution/economics metrics, PnL proof, deployment, promotion, or maker viability proof.
- Next auto-loop action after QA: create and execute `0615T008` small-cap live-test protocol / risk gate design and dry-run acceptance.

## 0615T008 QA Update

- `0615T008` has been created, business execution is complete, and QA is `已通过`.
- Scope: small-cap live-test protocol / risk gate design and dry-run acceptance after `0615T007` QA.
- It implemented `examples/binance_tick_mm/small_cap_live_test_protocol.py`, focused tests, design note, local dry-run artifacts, and business report.
- Official artifacts: `local_live_analysis/small_cap_live_test_protocol_0615T008/`.
- Final recommendation: `small_cap_live_test_protocol_ready_for_qa`.
- QA report: `.workflow/reports/0615T008-qa.md`; latest QA acceptance document now records `0615T008`.
- Protocol caps: `BTCUSDT`, `10` minutes, max gross notional `25 USDT`, max single order notional `5 USDT`, max position notional `10 USDT`, max loss `2 USDT`, maker-only/post-only required, default-on forbidden.
- `0615T008` does not open live, authorize credentials, connect endpoints, place/cancel orders, change strategy defaults, compute PnL proof, deploy, promote, or prove maker viability.
- Auto loop stops at the total-control approval boundary before any `0615T009` live window is opened.

## 0615T005 Prepared Task

- `0615T005` has been created and dispatched as the next formal task after `0615T004` QA passed.
- Scope: economics fee/rebate no-trading read-only source implementation based on accepted `0615T004` account inventory read-only source context and `0612T001` economics fee/rebate validator schema.
- It may implement only a task-local economics settlement input transform, redaction/opaque account and future-fill reference handling, local artifact writing, arithmetic/fail-closed safety gates, validation through `economics_fee_rebate_source.py`, tests, local artifacts, docs, and a business report.
- It must not implement or call endpoints, endpoint clients, signed requests, nonce handling, user streams, real private/order/account/live/economics data reads, remote execution, venue collection, runner consumption, order placement/cancellation/amendment, strategy/live/default-on/tiny-live behavior, real economics proof, real metrics, PnL proof, parameter search, deployment, promotion, or maker viability proof.
- Required first outputs are `examples/binance_tick_mm/economics_fee_rebate_read_only_source.py`, `examples/binance_tick_mm/test_economics_fee_rebate_read_only_source.py`, `docs/basis_positive_economics_fee_rebate_read_only_source.md`, `local_live_analysis/basis_positive_economics_fee_rebate_read_only_source_0615T005/**`, and `.workflow/reports/0615T005-business.md`.
- Business execution is complete and QA is `已通过`.
- It implemented a local-only no-trading economics fee/rebate source transform, redaction/opaque account and future-fill reference handling, local artifact writing, arithmetic/fail-closed checks, validation through `economics_fee_rebate_source.py`, focused tests, design note, local artifacts, and business report.
- Official artifacts: `local_live_analysis/basis_positive_economics_fee_rebate_read_only_source_0615T005/`.
- Final recommendation: `economics_fee_rebate_read_only_source_ready_for_qa`.
- QA report: `.workflow/reports/0615T005-qa.md`; latest QA acceptance document now records `0615T005`.
- Final recommendation taxonomy is `economics_fee_rebate_read_only_source_ready_for_qa` / `economics_fee_rebate_read_only_source_needs_revision` / `economics_fee_rebate_read_only_source_blocked`.

## 0615T004 Task Execution

- `0615T004` business execution is complete and QA is `已通过`.
- Scope: account inventory no-trading read-only source implementation based on accepted `0615T003` private order read-only collector context and `0611T004` account inventory validator schema.
- It implemented a local task-fixture to account inventory artifact transform, redaction/opaque account and future-order reference handling, local artifact writing, conservation/fail-closed safety gates, validation handoff through `account_inventory_source.py`, focused tests, a design note, local artifacts, and a business report.
- It did not implement or call endpoints, endpoint clients, signed requests, nonce handling, user streams, real private/order/account/live/economics data reads, remote execution, venue collection, runner consumption, order placement/cancellation/amendment, strategy/live/default-on/tiny-live behavior, inventory lifecycle proof, real metrics, PnL proof, parameter search, deployment, promotion, or maker viability proof.
- Required outputs were produced under `examples/binance_tick_mm/account_inventory_read_only_source.py`, `examples/binance_tick_mm/test_account_inventory_read_only_source.py`, `docs/basis_positive_account_inventory_read_only_source.md`, and `local_live_analysis/basis_positive_account_inventory_read_only_source_0615T004/**`.
- QA report: `.workflow/reports/0615T004-qa.md`; latest QA acceptance document now records `0615T004`.
- Final recommendation taxonomy is `account_inventory_read_only_source_ready_for_qa` / `account_inventory_read_only_source_needs_revision` / `account_inventory_read_only_source_blocked`.

## 0615T004 Prepared Task

- `0615T004` has been created and dispatched as the next formal task after `0615T003` QA passed.
- Scope: account inventory no-trading read-only source implementation based on accepted `0615T003` private order read-only collector context and `0611T004` account inventory validator schema.
- It may implement only a task-local account-state input transform, redaction/opaque account and order reference handling, local artifact writing, conservation/fail-closed safety gates, validation through `account_inventory_source.py`, tests, local artifacts, docs, and a business report.
- It must not implement or call endpoints, endpoint clients, signed requests, nonce handling, user streams, real private/order/account/live/economics data reads, remote execution, venue collection, runner consumption, order placement/cancellation/amendment, strategy/live/default-on/tiny-live behavior, inventory lifecycle proof, real metrics, PnL proof, parameter search, deployment, promotion, or maker viability proof.
- Required first outputs are `examples/binance_tick_mm/account_inventory_read_only_source.py`, `examples/binance_tick_mm/test_account_inventory_read_only_source.py`, `docs/basis_positive_account_inventory_read_only_source.md`, `local_live_analysis/basis_positive_account_inventory_read_only_source_0615T004/**`, and `.workflow/reports/0615T004-business.md`.
- Final recommendation taxonomy is `account_inventory_read_only_source_ready_for_qa` / `account_inventory_read_only_source_needs_revision` / `account_inventory_read_only_source_blocked`.

## 0615T003 Prepared Task

- `0615T003` has been created and dispatched as the next formal task after `0615T002` QA passed.
- Scope: private order response no-trading read-only collector implementation based on accepted `0615T002` boundary and `0611T002` validator schema.
- It may implement only a task-local input transform, redaction/opaque order reference handling, local artifact writing, fail-closed safety gates, validation through `private_order_response_source.py`, tests, local artifacts, docs, and a business report.
- It must not implement or call endpoints, endpoint clients, signed requests, nonce handling, user streams, real private/order/account/live/economics data reads, remote execution, venue collection, runner consumption, order placement/cancellation/amendment, strategy/live/default-on/tiny-live behavior, real metrics, PnL proof, parameter search, deployment, promotion, or maker viability proof.
- Required first outputs are `examples/binance_tick_mm/private_order_response_read_only_collector.py`, `examples/binance_tick_mm/test_private_order_response_read_only_collector.py`, `docs/basis_positive_private_order_response_read_only_collector.md`, `local_live_analysis/basis_positive_private_order_response_read_only_collector_0615T003/**`, and `.workflow/reports/0615T003-business.md`.
- Final recommendation taxonomy is `private_order_response_read_only_collector_ready_for_qa` / `private_order_response_read_only_collector_needs_revision` / `private_order_response_read_only_collector_blocked`.
- Business execution is complete and awaiting QA.
- It implemented a local-only no-trading collector transform, redaction/opaque order reference handling, local artifact writing, fail-closed forbidden-field checks, validation through `private_order_response_source.py`, focused tests, design note, local artifacts, and business report.
- Official artifacts: `local_live_analysis/basis_positive_private_order_response_read_only_collector_0615T003/`.
- Final recommendation: `private_order_response_read_only_collector_ready_for_qa`.

## 0615T002 Prepared Task

- `0615T002` has been created and dispatched as the next formal task.
- Scope: private order response read-only collector boundary / implementation design based on `0615T001` real-readiness convergence and the accepted `0611T002` local skeleton / `0610T006` source-line contract.
- It may design only endpoint/permission contract, schema field handoff into `private_order_response_source.py`, redaction/local-storage policy, no-trading safety gates, future implementation QA gates, next-task sequence, local artifacts, and a business report.
- It must not implement or use endpoints, credentials, signing, nonce handling, user streams, source collectors, private/order/account/live/economics data, remote execution, collection, runner consumption, order placement, order cancellation, strategy/live/default-on/tiny-live behavior, real execution metrics, real economics metrics, PnL proof, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Required first outputs are `docs/basis_positive_private_order_response_read_only_collector_boundary.md`, `local_live_analysis/basis_positive_private_order_response_read_only_collector_boundary_0615T002/**`, and `.workflow/reports/0615T002-business.md`.
- Final recommendation taxonomy is `private_order_response_read_only_collector_boundary_ready_for_qa` / `private_order_response_read_only_collector_boundary_needs_revision` / `private_order_response_read_only_collector_boundary_blocked`.
- This task does not consume the one allowed post-`0615T001` local-only exception because it is explicitly non-local in direction.
- Business execution is complete and awaiting QA.
- It produced the private order response read-only collector boundary design, endpoint/permission contract, field handoff mapping into `private_order_response_source.py`, redaction/storage policy, no-trading safety gates, future implementation QA gates, next-task sequence, manifest, boundary validation, and business report.
- Final recommendation: `private_order_response_read_only_collector_boundary_ready_for_qa`.
- Official artifacts: `local_live_analysis/basis_positive_private_order_response_read_only_collector_boundary_0615T002/`.
- It remains boundary/design only and does not authorize endpoint calls, endpoint clients, collector implementation, credentials/signing/nonce/user stream implementation, real private/order/account/live/economics data reads, runner consumption, order placement/cancellation/amendment, strategy/live/default-on/tiny-live behavior, real metrics, PnL proof, deployment, promotion, or maker viability proof.

## 0615T001 Prepared Task

- `0615T001` has been created and dispatched as the next formal task.
- Scope: design-only real-readiness / read-only collector boundary for the four accepted basis-positive execution source lines after the local skeleton chain (`0611T002`, `0611T003`, `0611T004`, `0612T001`) reached QA-passed readiness.
- It may produce only source-line real-readiness matrices, field-authority mapping, permission boundary taxonomy, runner consumption gate, next-task sequence, convergence policy, boundary validation, design doc, local artifacts, and business report.
- It must not implement or use endpoints, credentials, signing, nonce handling, user streams, source collectors, private/order/account/live/economics data, remote execution, collection, runner consumption, real execution metrics, real economics metrics, real fees/rebates/spread-capture proof, inventory lifecycle proof, PnL proof, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Required first outputs are `docs/basis_positive_execution_source_real_readiness_collector_boundary.md`, `local_live_analysis/basis_positive_execution_source_real_readiness_collector_boundary_0615T001/**`, and `.workflow/reports/0615T001-business.md`.
- Convergence policy: after `0615T001` QA, at most `1` additional local-only task may be dispatched, and only for a concrete blocker named by `0615T001`; otherwise the next execution-proof task must move to real source-line implementation or read-only collector work.
- Final recommendation taxonomy is `real_source_line_readiness_boundary_ready_for_qa` / `real_source_line_readiness_boundary_needs_revision` / `real_source_line_readiness_boundary_blocked`.
- Business execution is complete and QA is `已通过`.
- Final recommendation: `real_source_line_readiness_boundary_ready_for_qa`.
- QA report: `.workflow/reports/0615T001-qa.md`.
- Latest QA acceptance document now records `0615T001` as the latest effective QA result.

## 0612T001 QA Update

- `0612T001` has been created and dispatched as the next formal task.
- Scope: local-only economics fee/rebate settlement artifact skeleton / validator based on the accepted `0610T009` economics fee/rebate source-line contract and `0611T001` synthesis gate, with `0611T002` private-order, `0611T003` replay lifecycle, and `0611T004` account inventory local artifacts as context only.
- It may implement only task-scoped local economics/fee/rebate/spread-capture schema constants, fixture loader/parser, fail-closed validator, maker/taker classification checks, fee/rebate settlement checks, currency conversion / tick-value arithmetic validation, spread-capture consistency checks, settlement timestamp policy artifacts, CLI/help, tests, design note, local artifacts, and business report.
- It must not implement or use endpoints, credentials, signing, nonce handling, user streams, source collectors, economics/account/private/order/live data, remote execution, collection, runner consumption, real economics metrics, real fees/rebates/spread-capture proof, real execution metrics, PnL proof, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Required first outputs are `examples/binance_tick_mm/economics_fee_rebate_source.py`, `examples/binance_tick_mm/test_economics_fee_rebate_source.py`, `docs/basis_positive_economics_fee_rebate_source_artifact_skeleton.md`, `local_live_analysis/basis_positive_economics_fee_rebate_source_artifact_skeleton_0612T001/**`, and `.workflow/reports/0612T001-business.md`.
- Final recommendation taxonomy is `economics_fee_rebate_artifact_skeleton_ready_for_qa` / `economics_fee_rebate_artifact_skeleton_needs_revision` / `economics_fee_rebate_artifact_skeleton_blocked`.
- Business execution is complete and QA is `已通过`.
- It implements only local economics fee/rebate/spread-capture schema constants, fixture loader/parser, fail-closed validator, maker/taker classification checks, settlement checks, currency conversion / tick-value arithmetic validation, spread-capture consistency checks, timestamp policy artifacts, CLI/help, tests, design note, local artifacts, and business report.
- Official artifacts: `local_live_analysis/basis_positive_economics_fee_rebate_source_artifact_skeleton_0612T001/`.
- Final recommendation: `economics_fee_rebate_artifact_skeleton_ready_for_qa`.
- QA report: `.workflow/reports/0612T001-qa.md`.
- Latest QA acceptance document now records `0612T001` as the latest effective QA result.

## 0611T004 QA Update

- `0611T004` business execution is complete and QA is `已通过`.
- Scope: local-only account inventory artifact skeleton / validator based on the accepted `0610T008` account inventory source-line contract and `0611T001` synthesis gate, with `0611T002` private-order and `0611T003` replay lifecycle local artifacts as context only.
- It implements only task-scoped local account/inventory schema constants, fixture loader/parser, fail-closed validator, snapshot / transition / conservation checks, reconciliation boundary artifacts, CLI/help, tests, design note, local artifacts, and business report.
- It must not implement or use endpoints, credentials, signing, nonce handling, user streams, source collectors, account/private/order/live data, remote execution, collection, runner consumption, inventory lifecycle proof, realized inventory/exposure proof, real inventory metrics, real execution metrics, economics metrics, PnL proof, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Required first outputs are `examples/binance_tick_mm/account_inventory_source.py`, `examples/binance_tick_mm/test_account_inventory_source.py`, `docs/basis_positive_account_inventory_source_artifact_skeleton.md`, `local_live_analysis/basis_positive_account_inventory_source_artifact_skeleton_0611T004/**`, and `.workflow/reports/0611T004-business.md`.
- Final recommendation taxonomy is `account_inventory_artifact_skeleton_ready_for_qa` / `account_inventory_artifact_skeleton_needs_revision` / `account_inventory_artifact_skeleton_blocked`.
- Final recommendation: `account_inventory_artifact_skeleton_ready_for_qa`.
- Official artifacts: `local_live_analysis/basis_positive_account_inventory_source_artifact_skeleton_0611T004/`.
- QA report: `.workflow/reports/0611T004-qa.md`.
- Latest QA acceptance document now records `0611T004` as the latest effective QA result.

## 0611T003 QA Update

- `0611T003` business execution is complete and QA is `已通过`.
- Scope: local-only replay lifecycle validation / reconciliation gate based on the accepted `0610T007` replay lifecycle source-line contract and `0611T001` synthesis gate, with `0611T002` private-order local skeleton as context only.
- It may implement only task-scoped local lifecycle schema constants, fixture loader/parser, fail-closed validator, ordering/reconciliation policy artifacts, CLI/help, tests, design note, local artifacts, and business report.
- It must not implement or use endpoints, credentials, signing, nonce handling, user streams, source collectors, replay/live semantic implementation, private/order/account/live data, remote execution, collection, runner consumption, queue priority metrics, exact queue position proof, cancel-fill race metrics, real execution metrics, economics metrics, PnL proof, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Required first outputs are `examples/binance_tick_mm/replay_lifecycle_validation_gate.py`, `examples/binance_tick_mm/test_replay_lifecycle_validation_gate.py`, `docs/basis_positive_replay_lifecycle_validation_reconciliation_gate.md`, `local_live_analysis/basis_positive_replay_lifecycle_validation_gate_0611T003/**`, and `.workflow/reports/0611T003-business.md`.
- Final recommendation taxonomy is `replay_lifecycle_validation_gate_ready_for_qa` / `replay_lifecycle_validation_gate_needs_revision` / `replay_lifecycle_validation_gate_blocked`.
- Final recommendation: `replay_lifecycle_validation_gate_ready_for_qa`.
- Official artifacts: `local_live_analysis/basis_positive_replay_lifecycle_validation_gate_0611T003/`.
- QA report: `.workflow/reports/0611T003-qa.md`.
- Latest QA acceptance document now records `0611T003` as the latest effective QA result.

## 0611T002 QA Update

- `0611T002` business execution is complete and QA is `已通过`.
- Scope: local-only `private_order_response` artifact skeleton / validator based on the accepted `0610T006` source-line contract and `0611T001` synthesis gate.
- It may implement only task-scoped local schema constants, fixture loader/parser, fail-closed validator, CLI/help, tests, design note, local artifacts, and business report.
- It must not implement or use endpoints, credentials, signing, nonce handling, user streams, source collectors, private/order/account/live data, remote execution, collection, runner consumption, real execution metrics, economics metrics, PnL proof, strategy/live/default-on/tiny-live behavior, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Required first outputs are `examples/binance_tick_mm/private_order_response_source.py`, `examples/binance_tick_mm/test_private_order_response_source.py`, `docs/basis_positive_private_order_response_source_artifact_skeleton.md`, `local_live_analysis/basis_positive_private_order_response_source_artifact_skeleton_0611T002/**`, and `.workflow/reports/0611T002-business.md`.
- Final recommendation taxonomy is `private_order_response_artifact_skeleton_ready_for_qa` / `private_order_response_artifact_skeleton_needs_revision` / `private_order_response_artifact_skeleton_blocked`.
- Final recommendation: `private_order_response_artifact_skeleton_ready_for_qa`.
- Official artifacts: `local_live_analysis/basis_positive_private_order_response_source_artifact_skeleton_0611T002/`.
- QA report: `.workflow/reports/0611T002-qa.md`.
- Latest QA acceptance document now records `0611T002` as the latest effective QA result.

## 0611T001 QA Update

- `0611T001` business execution is complete and QA is `已通过`.
- Scope: design-only source-line synthesis / implementation-readiness gate over the accepted `0610T006` / `0610T007` / `0610T008` / `0610T009` source-line contracts and `0610T005` decomposition.
- Required coverage includes source-line contract registry, implementation-readiness gate matrix, source dependency reconciliation matrix, forbidden overclaim matrix, next-task sequence, manifest, boundary validation, design document, and business report.
- It may recommend future scoped implementation tasks, but it must not authorize endpoint/source reader/collector/runner implementation inside `0611T001`.
- It must preserve current proof rejection for all seven execution gaps, PnL, maker execution viability, live/default-on/tiny-live readiness, deployment, and promotion.
- It does not authorize endpoint implementation, source reader/collector implementation, runner implementation, private/order/account/live data use, user stream, signing/nonce handling, real execution metrics, real economics metrics, PnL proof, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Official artifacts: `local_live_analysis/basis_positive_execution_source_line_synthesis_gate_0611T001/`.
- Synthesis gate design: `docs/basis_positive_execution_source_line_synthesis_gate.md`.
- Final recommendation: `source_line_synthesis_gate_ready_for_qa`, meaning only that the synthesis/gate design is ready for QA/controller review.
- QA report: `.workflow/reports/0611T001-qa.md`.
- Latest QA acceptance document now records `0611T001` as the latest effective QA result.

## 0610T009 QA Update

- `0610T009` business execution is complete and QA is `已通过`.
- Scope: design-only `economics_fee_rebate_source_line` contract.
- It may cover only `fees_rebates_spread_capture` as the primary gap.
- Required contract coverage includes economics/fee/rebate/spread-capture artifact schema, fee/rebate settlement taxonomy, spread-capture taxonomy, maker/taker classification policy, currency conversion / tick-value policy, settlement timestamp policy, reconciliation boundary, validation gates, and overclaim reject rules.
- Inputs are restricted to accepted local `0610T008` / `0610T007` / `0610T006` / `0610T005` / `0610T004` / `0610T003` / `0610T002` design artifacts, manifests, and QA/business reports.
- `0610T006` private-order response artifacts may be used only as future fill dependency context, `0610T007` replay lifecycle artifacts only as future timestamp/order consistency context, and `0610T008` account inventory artifacts only as future reconciliation context.
- It must explicitly reject hypothetical spread, fill notional, order fills alone, public markout alone, account inventory alone, or replay lifecycle alone as proof of fees/rebates/spread capture or PnL.
- It does not authorize economics endpoint implementation, source reader/collector implementation, runner implementation, private/order/account/live data use, user stream, signing/nonce handling, real economics metrics, real execution metrics, PnL proof, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- Official artifacts: `local_live_analysis/basis_positive_economics_fee_rebate_source_line_contract_0610T009/`.
- Source-line contract: `docs/basis_positive_economics_fee_rebate_source_line_contract.md`.
- Final recommendation: `economics_fee_rebate_contract_ready_for_qa`, meaning only that the design contract is ready for QA/controller review.
- QA report: `.workflow/reports/0610T009-qa.md`.
- Latest QA acceptance document now records `0610T009` as the latest effective QA result.

## 0610T007 / 0610T008 QA Update

- `0610T007` and `0610T008` were executed in parallel as design-only contracts.
- `0610T007` business execution is complete and QA is `已通过`.
- `0610T007` official artifacts: `local_live_analysis/basis_positive_replay_lifecycle_semantics_source_line_contract_0610T007/`.
- `0610T007` source-line contract: `docs/basis_positive_replay_lifecycle_semantics_source_line_contract.md`.
- `0610T007` final recommendation: `replay_lifecycle_contract_ready_for_qa`.
- `0610T007` covers only `queue_priority` and `cancel_fill_race` as future design labels.
- `0610T007` commits: `139b76a` (`0610T007 replay lifecycle contract`) and `692f445` (`0610T007 business report metadata`).
- `0610T008` business execution is complete and QA is `已通过`.
- `0610T008` official artifacts: `local_live_analysis/basis_positive_account_inventory_source_line_contract_0610T008/`.
- `0610T008` source-line contract: `docs/basis_positive_account_inventory_source_line_contract.md`.
- `0610T008` final recommendation: `account_inventory_contract_ready_for_qa`.
- `0610T008` covers only `inventory_lifecycle` as a future design label and explicitly states that order fills alone cannot prove inventory lifecycle.
- `0610T008` commits: `66127b7` (`0610T008 account inventory contract`) and `156f6da` (`0610T008 business report metadata`).
- Parallel write-scope check passed: the business-thread commits for T007/T008 touched only their own task/report/doc/artifact paths and did not include shared tracking files.
- QA reports: `.workflow/reports/0610T007-qa.md` and `.workflow/reports/0610T008-qa.md`.
- Latest QA acceptance document now records `0610T008` as the latest effective QA result.
- Neither task authorizes source reader/collector implementation, runner implementation, private/order/account/live endpoint use, user stream, signing/nonce handling, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T008 Prepared Task

- `0610T008` has been created as a prepared parallel-eligible design-only task.
- Scope: design-only `account_inventory_source_line` contract.
- It may cover only `inventory_lifecycle` as the primary gap.
- Required contract coverage includes account/inventory artifact schema, inventory snapshot / transition taxonomy, conservation checks, reconciliation boundary, fail-closed gates, and explicit rejection that order fills alone can prove inventory lifecycle.
- Inputs are restricted to accepted local `0610T006` / `0610T005` / `0610T004` / `0610T003` / `0610T002` design artifacts, manifests, and QA/business reports.
- `0610T006` private order response artifacts may be used only as future transition input / future cross-check context, not current inventory lifecycle proof.
- It does not authorize account endpoint implementation, source reader/collector implementation, runner implementation, private/order/account/live data use, user stream, signing/nonce handling, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- It is parallel-eligible with `0610T007` because both are design-only contracts, their primary gaps and output paths are disjoint, and neither business thread may modify shared tracking files (`task_plan.md`, `progress.md`, `findings.md`, `docs/qa-acceptance-report.md`). Shared tracking must be updated later by total control in a single serial step.

## 0610T006 Business Update

- `0610T006` business execution is complete and QA is `已通过`.
- It created the design-only `private_order_response_source_line` contract after `0610T005` QA.
- Official artifacts: `local_live_analysis/basis_positive_private_order_response_source_line_contract_0610T006/`.
- Source-line contract: `docs/basis_positive_private_order_response_source_line_contract.md`.
- Final recommendation: `private_order_response_contract_ready_for_qa`.
- It covers only `fill_probability`, `post_only_reject_behavior`, and `real_order_lifecycle` as future design labels.
- It produced artifact schema, response/reject/lifecycle taxonomies, timestamp policy, validation gates, overclaim reject rules, manifest, boundary validation, and a business report.
- It does not authorize endpoint implementation, source reader/collector implementation, runner implementation, private/order/account/live data use, user stream, signing/nonce handling, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T007 Prepared Task

- `0610T007` has been created and dispatched as the next formal task.
- Scope: design-only `replay_lifecycle_semantics_source_line` contract after `0610T006` QA.
- It may cover only `queue_priority` and `cancel_fill_race` as future design labels.
- Inputs are restricted to accepted local `0610T006` / `0610T005` / `0610T004` / `0610T003` / `0610T002` design artifacts, manifests, and QA/business reports.
- Required outputs are the replay lifecycle source-line design doc, replay lifecycle event schema, queue boundary matrix, cancel/fill race ordering policy, timestamp policy, replay/live proof-limit rules, validation gates, overclaim reject rules, manifest, boundary validation, and business report.
- It does not authorize replay/live semantic implementation, source reader/collector implementation, runner implementation, private/order/account/live data use, user stream, signing/nonce handling, exact queue position proof, cancel-fill race metric proof, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.
- It is parallel-eligible with `0610T008` because both are design-only contracts, their primary gaps and output paths are disjoint, and neither business thread may modify shared tracking files. Shared tracking must be updated later by total control in a single serial step.

## 0610T006 Prepared Task

- `0610T006` has been created, dispatched, executed, and accepted by QA.
- Scope: design-only `private_order_response_source_line` contract after `0610T005` QA.
- It may cover only `fill_probability`, `post_only_reject_behavior`, and `real_order_lifecycle` as future design labels.
- It must define response artifact schema, response/reject/lifecycle label taxonomy, timestamp policy, terminal-state consistency, fail-closed validation gates, and overclaim rejection rules.
- It does not authorize private/order endpoint implementation, source reader/collector implementation, runner implementation, private/order/account/live data use, user stream, signing/nonce handling, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T005 QA Update

- `0610T005` QA is `已通过`.
- It split the seven execution gaps into four source-design lines using truth authority, label unit, causal time semantics, permission boundary, validation oracle, and overclaim failure mode.
- Official artifacts: `local_live_analysis/basis_positive_execution_source_design_decomposition_0610T005/`.
- Source decomposition design: `docs/basis_positive_execution_source_design_decomposition.md`.
- Final recommendation: `private_order_source_design_ready_next`.
- This means only that a later separately dispatched design-only task may define the `private_order_response_source_line` contract.
- It does not authorize source implementation, runner implementation, private/order/account/live endpoint use, user stream, signing/nonce handling, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T005 Business Update

- `0610T005` business execution is complete and QA is `已通过`.
- It split the seven execution gaps into source-design lines using truth authority, label unit, causal time semantics, permission boundary, validation oracle, and overclaim failure mode.
- Official artifacts: `local_live_analysis/basis_positive_execution_source_design_decomposition_0610T005/`.
- Source decomposition design: `docs/basis_positive_execution_source_design_decomposition.md`.
- Final recommendation: `private_order_source_design_ready_next`.
- It produced design docs, matrices, dependency graph, boundary validation, manifest, next-task sequence, and a business report.
- It does not authorize source implementation, runner implementation, private/order/account/live endpoint use, user stream, signing/nonce handling, real execution metrics, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T004 QA Update

- `0610T004` QA is `已通过`.
- It implemented a local fail-closed/read-only execution-evidence runner skeleton over accepted `0610T003`/`0610T002` contract and gate artifacts.
- Official artifacts: `local_live_analysis/basis_positive_execution_evidence_fail_closed_runner_0610T004/`.
- Final recommendation: `fail_closed_runner_skeleton_ready_for_qa`.
- It emits proof-limited unavailable status rows for all seven execution gaps under current sources.
- It does not authorize real execution metrics, private/order/account/live data use, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T003 Business Update

- `0610T003` business execution is complete and QA is `已通过`.
- It created the design/gate-only source availability and runner implementation gate over QA-passed `0610T002` artifacts.
- Required outputs are `docs/basis_positive_execution_evidence_source_gate.md` and `local_live_analysis/basis_positive_execution_evidence_source_gate_0610T003/`.
- Final recommendation: `runner_skeleton_ready_with_fail_closed_sources`.
- It classified all seven execution gaps as fail-closed placeholder only under current sources; actual execution-proof metrics remain blocked by private/order response, replay/lifecycle semantics, account/inventory, or economics source-design requirements.
- It does not authorize runner implementation, private/order, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T002 Business Update

- `0610T002` business execution is complete and QA is `已通过`.
- It created the design-only read-only basis-positive execution-evidence runner contract.
- Official artifacts: `local_live_analysis/basis_positive_execution_evidence_runner_contract_0610T002/`.
- Runner contract: `docs/basis_positive_execution_evidence_runner_contract.md`.
- Final recommendation: `read_only_execution_evidence_runner_design_ready`.
- This means only that the current runner contract/design artifacts are ready for QA/controller review. It does not indicate implementation readiness and does not authorize runner implementation, private/order, strategy/live/default-on/tiny-live, case-library/shadow decisions, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0610T001 Business Update

- `0610T001` business execution is complete and QA is `已通过`.
- It created the design-only basis-positive execution-evidence requirements contract.
- Official artifacts: `local_live_analysis/basis_positive_execution_evidence_requirements_0610T001/`.
- Design contract: `docs/basis_positive_execution_evidence_requirements.md`.
- Final recommendation: `execution_evidence_runner_contract_ready`.
- This means only that a later separately scoped read-only runner contract/design task can be considered after QA. It does not authorize runner implementation, case-library implementation, shadow decisions, strategy/private/order/live/default-on/tiny-live behavior, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0609T011 Business Update

- `0609T011` business execution is complete and is now `待验收`.
- It implemented the local read-only proxy evidence synthesis runner over QA-passed T010 artifacts.
- Official artifacts: `local_live_analysis/basis_positive_proxy_evidence_synthesis_0609T011/`.
- Final recommendation: `continue_to_execution_evidence_design`.
- This means only that a later separately scoped design task can define execution-layer evidence requirements. It does not authorize case-library implementation, source-row case catalogs, shadow decisions, executable triggers, strategy/private/order/live/default-on/tiny-live behavior, parameter search, deployment, promotion, or execution-layer maker viability proof.

## 0609T010 Prepared Task

- `0609T010` QA has passed.
- It implemented the local read-only maker-viability proxy runner over T008/T009 allowlisted artifacts.
- Official artifacts: `local_live_analysis/basis_positive_maker_viability_proxy_0609T010/`.
- Final recommendation: `read_only_proxy_evidence_ready_for_qa`.
- It does not authorize case-library implementation, source-row case catalogs, shadow decisions, executable triggers, strategy/private/order/live/default-on/tiny-live, parameter search, deployment recommendation, promotion, or execution-layer maker viability proof.

## 0609T009 Prepared Task

- `0609T009` business execution is complete and is now `待验收`.
- It created the design/contract artifacts for `Basis-positive clean context execution-evidence gap planning / read-only maker-viability proxy contract`.
- Final recommendation: `read_only_proxy_runner_ready_for_implementation`.
- Scope remained design/contract only: translate `0609T008` row-level read-only artifacts into a later read-only maker-viability proxy runner contract.
- It does not authorize proxy runner implementation, case-library implementation, source-row case catalogs, shadow decisions, strategy/private/order/live/default-on/tiny-live, parameter search, deployment recommendation, promotion, or execution-layer maker viability proof.

## 0609T001 Business Update

- `0609T001` business execution is complete and is now `待验收`.
- New runner: `examples/hyperliquid/canonical_basis_positive_wrong_way_decomposition.py`.
- Focused tests: `examples/hyperliquid/test_canonical_basis_positive_wrong_way_decomposition.py`.
- Official artifacts: `local_live_analysis/canonical_basis_positive_wrong_way_decomposition_0609T001/`.
- Final recommendation: `targeted_collection_ready`. This only means a later separately scoped collection task can be designed; T001 collected no new data and does not authorize strategy implementation, private/order endpoints, order lifecycle, case-library, shadow decisions, live/default-on/tiny-live, parameter search, or promotion.

## Current Focus

- Use `workflow-kit` and the local dashboard as the persistent development workflow for the hftbacktest Binance maker MM work.
- Latest accepted robustness source: `0604T003` has passed QA and makes event-mode + de-aliased future-row-delta evidence canonical while downgrading ordinary synthetic fixed-grid short-horizon alias evidence to diagnostic-only.
- Latest accepted workflow fact source: `0605T004` passed QA as the shutdown final proof no-order dry-run validation. It proves local fake no-network proof/log/audit observability only; it does not prove a real exchange shutdown run and does not authorize live/default-on/tiny-live or promotion.
- `0608T001` has reconciled workflow tracking so the controller no longer treats `0604T015` as the active task.
- Current implementation focus is no longer `0604T015`. The shutdown safety chain has moved through `0604T016` failed QA, `0605T001` partial-fill local proof repair, `0605T002` local-only proof diagnosis, `0605T003` exchange-reconciliation/final-proof hardening, and `0605T004` fake no-order dry-run validation.
- T003 now constrains the follow-up path: after QA, retain only a narrow Step 5C default-off / diagnostic-first quote-anchor safety layer. It should not repair audit_depth/bookTicker/top5 row-exact drift, promote top5 to hard anchor, redesign quote placement, change replay lifecycle, or start live.
- `0518T004` has completed business-thread execution and passed QA.
- `0519T001` QA passed; `0519T002` QA passed and closed Step 6 for roadmap progression.
- `0519T003` QA passed and closed Step 7 as a design-only inventory / execution model contract.
- `0519T004` completed Step 8 design-only quote-update / API-limit hygiene and passed QA.
- `0519T005` completed the narrow Step 8B read-only diagnostic / implementation-planning task and passed QA.
- `0519T006` passed QA as Step 8C default-off quote-update helper / instrumentation implementation.
- `0519T007` passed QA as the Step 9A design-only task.
- `0519T008` passed QA as the Step 9B default-off offline runner implementation task.
- `0519T009` passed QA as a narrow current-format live control collection plus T008 rerun task.
- `0519T010` passed QA as the planning-only Step 9C task.
- `0519T011` completed business-thread execution and passed QA.
- `0520T001` QA 已通过 as the read-only Step 9C multi-sample validation task.
- `0520T002` has been created as the narrow Step 9C runner/artifact hardening follow-on and is now `已通过`.
- `0521T001` has completed live collection, replay, and archive work and has passed QA.
- `0521T002` has completed the read-only Step 9C candidate x scenario bucket multi-sample determination task and passed QA.
- `0525T001` has completed business execution as the current Step 9D fine-bucket refinement task and passed QA.
- `0529T004` has completed the Hyperliquid read-only public market-data evidence hardening task and passed QA.
- `0529T005` has completed the Binance Stage 9L read-only rejection decomposition / bucket coarsening task and passed QA.
- `0530T001` has passed QA as a Hyperliquid design-only/read-only public market-data research consumer contract task.
- `0530T002` has passed QA after total controller explicitly ratified / accepted the already collected Stage 9M artifact.
- `0531T001` has passed QA as the Hyperliquid read-only consumer implementation task.
- `0531T002` has been created as the next Binance Stage 9N read-only evidence viability refinement task, is unblocked by `0530T002` QA, has completed business execution, and has passed QA.
- `0601T001` has passed QA as the Hyperliquid lag-venue public BTC sample collection and alignment task.
- `0602T001` has passed QA as the synchronized public-only Binance lead / Hyperliquid lag collection task.
- `0601T002` has passed QA as the read-only Binance lead / Hyperliquid lag as-of joined-feature input task.
- `0601T003` has passed QA as the read-only Binance-to-Hyperliquid lead-lag stability analyzer task.
- `0601T004` has passed QA as the Binance-led Hyperliquid maker data input contract.
- `0601T005` has passed QA as the Binance-led Hyperliquid read-only pricing-signal runner implementation.
- `0601T006` has passed QA as public-only collection / initial synthetic-grid aggregate evidence; its formal robustness interpretation is superseded by `0604T003` canonical event-mode artifacts.
- `0604T001`, `0604T002`, and `0604T003` have passed QA. Future Binance-led Hyperliquid pricing-signal robustness decisions should use the `0604T003` canonical event-mode artifacts.
- `0604T009` has passed QA and contributes one read-only Milestone 3 executability candidate regime: `regime_011_1000_spread_10_20_ticks`. It does not authorize maker action, strategy, live/default-on/tiny-live, parameter search, or promotion.
- `0604T015` / `0604T016` are historical shutdown-proof intermediate tasks. `0604T016` QA failed, and its defect chain was closed by `0605T001` / `0605T003` / `0605T004`.
- Latest completed milestones: `0515T003` QA 已通过，`0516T001` QA 已通过，`0516T002` QA 已通过，`0518T001` QA 已通过，`0518T002` QA 已通过，`0518T003` QA 已通过，`0518T004` QA 已通过。

## Current Status

- Workflow files: initializing.
- Active task: `0615T002`
- Active task status: `待执行`
- Parallel condition result: both business threads avoided shared tracking writes; total control has updated tracking after both business reports became available.
- Latest business result awaiting QA: 无
- Latest QA source of truth: `0615T001` (`已通过`)
- Latest prepared next task: `0615T002`
- Latest workflow housekeeping: `0608T001` (`已通过`, no QA)
- Prepared independent task: `0530T001` (`已通过`)
- Prepared Binance task: `0530T002` (`已通过`)
- Prepared Hyperliquid task: `0531T001` (`已通过`, unblocked by `0530T001` QA)
- Prepared Binance follow-up: `0531T002` (`已通过`, unblocked by `0530T002` QA)
- Prepared Hyperliquid lag sample: `0601T001` (`已通过`)
- Prepared synchronized cross-exchange sample: `0602T001` (`已通过`)
- Prepared cross-exchange join: `0601T002` (`已通过`)
- Prepared lead-lag stability analyzer: `0601T003` (`已通过`)
- Prepared read-only pricing-signal runner: `0601T005` (`已通过`, unblocked by `0601T004` QA)
- Prepared multi-sample robustness validation: `0601T006` (`已通过`, superseded for formal robustness interpretation by `0604T003`)
- Prepared horizon-alias canonicalization repair: `0604T003` (`已通过`, canonical event-mode robustness source)
- Prepared canonical evidence loader foundation: `0604T004` (`已通过`, shared foundation for subsequent Milestone 0 / Milestone 1 workers)
- Prepared canonical evidence source lock / guard hardening: `0604T005` (`已通过`, canonical source-lock guard foundation)
- Prepared canonical signal quality ranking: `0604T006` evidence is accepted for downstream use after `0604T008` repaired the report bucket-consistency QA defect.
- Prepared canonical horizon / regime diagnostics: `0604T007` (`已通过`, parallel Milestone 1 worker)
- Prepared T006 report consistency repair: `0604T008` (`已通过`)
- Prepared canonical signal + horizon/regime synthesis: `0604T009` (`已通过`, one read-only Milestone 3 executability candidate)
- Prepared Milestone 3 maker executability assessment: `0608T002` (`已通过`, final recommendation `reject_not_maker_executable`)
- Regime 011 directional momentum viability assessment: `0608T003` (`已通过`, final recommendation `reject_directional_edge_unstable`)
- Regime 011 feature-conditioned validity diagnosis: `0608T004` (`已通过`, final recommendation `watch_needs_contract_visibility_clarification`, no supported/watch-valid patterns)
- Regime 011 basis-context visibility / lineage diagnosis: `0608T005` (`已通过`, final contract decision `upgrade_to_context_only_supported` for `context_basis_mid_ticks > 0` as read-only context only)
- Basis-positive independent robustness diagnosis outside Regime 011: `0608T006` (`待验收`, final recommendation `needs_more_samples`, read-only only)
- Historical live shutdown bounded-wait fix: `0604T013` (`作废` as an open queue item; superseded by later proof-semantics tasks)
- Historical live shutdown wait-result ambiguity diagnosis: `0604T015` (`作废` as an open queue item; diagnosis consumed by later repair chain)
- Historical live shutdown cancel proof semantics fix: `0604T016` (`未通过`; defect repaired by `0605T001` and later proof hardening)
- Shutdown partial-fill local proof repair: `0605T001` (`已通过`)
- Shutdown local-only proof diagnosis: `0605T002` (`已通过`)
- Shutdown exchange-reconciliation/final-proof hardening: `0605T003` (`已通过`)
- Shutdown final proof no-order dry-run validation: `0605T004` (`已通过`, latest shutdown/live-safety QA fact source)
- Workflow current-state reconciliation: `0608T001` (`已通过`, no QA; tracking cleanup only)
- Current blocker: none.

## Next Task

- Controller state cleanup `0608T001` reconciles the stale queue created by `0604T013` / `0604T015` / `0604T016`. After this cleanup, do not dispatch from the old `0604T015` pending state.
- If continuing shutdown/live-safety work, the next task must be a separate, explicitly scoped real-environment proof design or no-order observation task. `0605T004` was fake/no-network only and does not authorize live/default-on/tiny-live/promotion.
- `0608T003` passed QA with `reject_directional_edge_unstable`. `0608T004` passed QA with no supported/watch-valid feature-conditioned case pattern. `0608T005` passed QA with `upgrade_to_context_only_supported` for `context_basis_mid_ticks > 0` as read-only context only. `0608T006` business execution is complete and awaiting QA with `needs_more_samples`: broad basis-positive evidence is positive but sample-concentrated and cost/tail-caveated. Regime 011 and basis-positive context still should not progress to maker/directional case-library, shadow decisions, strategy implementation, private/order endpoints, live/default-on/tiny-live, parameter search, or promotion.
- `0526T007` now has a complete derived chain on `5-26-active-makeredge-control-180min-a`: T009 sidecar/join, Stage 5, Step 5C, Stage 6, Step 9B, and Step 9D are present. It remains a current-format no-rule/default-off control sample only.
- `0526T008` has passed QA as a 30min current-format no-rule/default-off control run with run id `5-26-active-minmove-control-30min-b`. It remains diagnostic/control data only and does not authorize candidate enablement, guard relaxation, parameter sweep, tiny live, default-on, or promotion.
- `0526T006` passed QA. Its output is a design-only `inventory_aware_quote_placement_request` contract and a later read-only runner contract; it did not implement strategy behavior, run sweep, start live, or make promotion claims.
- `0528T001` passed QA. It produced Stage 9I read-only inventory-aware quote-placement artifacts and a `reject` recommendation; it does not authorize strategy behavior changes, parameter search, live/default-on, guard relaxation, or promotion.
- `0527T001` passed QA. It found the `0526T008` bloat is primarily replay audit lifecycle export / order-state tracking repetition in audit replay mode, not a Stage 6 label-generation problem; repeated `cancel_ack` rows are redundant for Stage 6 after keeping first terminal lifecycle fact per order.
- `0528T002` passed QA. It implements compact replay lifecycle audit export for Stage 6 input, with terminal lifecycle rows de-duplicated inside the compact artifact by Stage 6 semantics. Bounded `0526T008` verification shows the 21GB full-prefix bloat path drops `250,000` scanned rows to `23,345` compact rows by skipping `226,655` duplicate terminal rows, and Stage 6 consumes `audit_bt_audit_replay.compact_lifecycle.csv` by contract. It does not authorize strategy, live, parameter, fill/cancel semantic, guard, default-on, or promotion changes.
- `0529T001` passed QA. It defines the next Binance maker policy direction as fill-quality-first and recommends a read-only fill-quality bucket synthesis runner before any new strategy policy implementation. It rejects continuing the fixed `0528T001` inventory skeleton, the current `0526T004` min-move grid, tiny-live/default-on/promotion, and more audit-bloat work unless a regression appears.
- `0529T002` passed QA. Stage 9K found `0` Shape A candidates and `0` Shape B candidates; clean-only decision-visible trigger buckets are mostly `reject_quality_negative` (`246`) or `needs_more_clean_fills` (`68`), so no policy-design follow-up is supported yet.
- `0529T004` passed QA. It collected a fresh 120s public-only Hyperliquid BTC sample, generated subscription/session/recovery evidence, ran the existing alignment runner, and classified the sample as `passes_pricing_research_market_view`. This remains public market-data-only evidence and does not authorize private connector, order lifecycle, strategy live logic, parameter search, default-on, tiny-live, or promotion.
- `0529T005` passed QA. It implemented the read-only Stage 9L runner, generated artifacts under `local_live_analysis/stage9l_fill_quality_rejection_decomposition_0529T005/`, and produced final classification `needs_targeted_clean_fills`. Shape A / Shape B candidates remain `0`; churn-warning sensitivity alone creates no ready candidates. It does not authorize strategy implementation, live/default-on, parameter search, guard relaxation, tiny-live, or promotion.
- `0530T001` passed QA. It wrote `docs/hyperliquid_public_market_data_research_consumer_design.md`, uses `0529T004` as the fresh public-sample entry point, rechecked official Hyperliquid public docs successfully, and recommends only a later read-only local consumer implementation. QA rechecked the official public docs URLs and received HTTP 200. It continues to forbid private connector, order lifecycle, strategy live, parameter search, default-on, tiny-live, and promotion.
- `0530T002` passed QA after total controller ratified / accepted the already collected Stage 9M artifact. The data/artifact chain is reproducible: existing accepted current-format samples were scanned first; no existing sample could add top-gap evidence; one `120min` no-rule/default-off control sample was collected as `5-31-stage9m-cleanfill-control-120min-a`; maker acceptance and market-view passed; T009 join quality is clean; Stage 5 has `5742` submits / `106` fills; Stage 6 is `methodology_valid_single_sample`; and Stage 9K/9L were rerun. Aggregate clean-only Stage 9K fills increased `994 -> 1102`, but the top Stage 9L gap only improved `34 -> 36` fills and remains below threshold. Stage 9L final classification is still `needs_targeted_clean_fills`, coarsened ready bucket count is `0`, Shape A / Shape B candidate rows are `0`, and policy design remains blocked. Caveat: original fixed logs/reports did not prove pre-start approval; acceptance is based on current controller ratification.
- `0531T001` passed QA. It implements only a read-only local artifact consumer over `0529T004`, producing market-view time series, pricing features, quality summaries, and a recommendation markdown. It does not authorize fresh collection, private connector, order lifecycle, strategy live, parameter search, default-on, tiny-live, or promotion.
- `0531T002` is now prepared as the next Binance formal task and is unblocked by `0530T002` QA. It should read only existing Stage 9M/9K/9L artifacts, explain why the new sample added `+108` aggregate fills but only `+2` top-gap fills, estimate collection duration for the `36 -> 40` and `+20` top-gap targets, and recommend whether to stop, briefly continue for threshold-crossing only, or pivot to alternative read-only regime refinement. It does not authorize collection, policy design, strategy implementation, candidate enablement, guard relaxation, parameter search, tiny-live/default-on, or promotion.
- `0601T001` passed QA. Its Hyperliquid public-only lag-venue sample reached `passes_pricing_research_market_view` and remains venue-state / execution-context evidence only.
- `0602T001` passed QA. Its synchronized public-only sample has `1800.105s` overlap and can feed a later read-only `0601T002` join, but it does not establish a Binance-lead / Hyperliquid-lag statistical effect.
- `0601T002` passed QA. It generated read-only joined features over the synchronized public sample with `primary_usable_row_count=3596`, `future_join_count=0`, and trade pressure disabled.
- `0601T003` passed QA. It generated read-only lead-lag stability evidence with verdict counts `18 stable / 6 watch / 30 unstable`, and it explicitly records effective future age because the current Hyperliquid decision cadence is roughly 500ms.
- `0601T004` passed QA. The accepted next boundary is a later read-only pricing-signal runner only; primary Binance lead allowlist is `binance_top5_imbalance`, `binance_microprice_minus_mid_ticks`, `binance_mid_move_ticks_from_prev`, and `binance_top5_bid_qty`. It does not authorize strategy implementation, private/order endpoints, live/default-on/tiny-live, parameter search, or promotion.
- `0601T005` passed QA. The read-only runner generated `21541` pricing signal rows, `4` feature quality rows, `30` horizon label summary rows, `540` feature/regime rows, and `54` venue-state conditioning rows from `3596` primary rows. Recommendation is `keep_for_read_only_research` with `single_public_sample_caveat=true`; this remains public-artifact read-only research only and does not authorize strategy/private/order/live/parameter/default-on/tiny-live/promotion.
- `0601T006` has passed QA as the public-only collection / initial aggregate step. Its ordinary synthetic fixed-grid aggregate remains diagnostic-only after `0604T003`.
- `0604T003` has passed QA and is the formal robustness source: canonical event-mode aggregate has `canonical_sample_count=3` and recommendation `continue_read_only_runner_refinement`; ordinary synthetic `xemm_0603_quiet_a/b/c` comparison has `canonical_sample_count=0`, `diagnostic_synthetic_sample_count=3`, and recommendation `needs_more_public_samples`.
- `0604T004` passed QA. It implemented the narrow canonical event-mode evidence loader / validator foundation before any parallel Milestone 0 / Milestone 1 runners. The accepted `0604T003` canonical aggregate validates to `canonical_sample_count=3`, and the synthetic diagnostic comparison validates to `canonical_sample_count=0` with `diagnostic_rejection_count=3`. It did not perform signal ranking, regime selection, strategy implementation, private/order endpoints, live/default-on/tiny-live, parameter search, or promotion.
- `0604T005` passed QA. It added reusable canonical source-lock guard checks over the `0604T003` canonical event-mode aggregate through the `0604T004` foundation, produced task-scoped source-lock artifacts, accepted canonical evidence with `canonical_sample_count=3`, and negative-validated the synthetic diagnostic comparison with `canonical_sample_count=0` / `diagnostic_rejection_count=3`.
- `0604T006` completed read-only canonical signal quality ranking over the four `0601T004` allowlist features through the `0604T005` guard path. Its first QA found a report bucket-consistency defect, but `0604T008` repaired that defect and QA passed; downstream work should use the T008-refreshed artifacts under `local_live_analysis/canonical_signal_quality_ranking_0604T006/`. Current ranking remains: `binance_mid_move_ticks_from_prev=keep_for_read_only_research`; `binance_top5_imbalance`, `binance_top5_bid_qty`, and `binance_microprice_minus_mid_ticks=watch_regime_dependent`.
- `0604T007` passed QA as read-only canonical horizon / regime diagnostics over the `0604T003` event-mode aggregate through the `0604T004` loader path. It classified `100/250ms` horizons as watch-only, kept `500/1000/5000/10000ms` as diagnostic-supported only, and left all regime buckets watch-only with no final regime selection, strategy, private/order, live/default-on/tiny-live, parameter search, or promotion.
- `0604T008` passed QA as the narrow T006 report-consistency repair. It did not change ranking scoring, allowlist, source-lock guard, canonical loader, strategy, private/order, live/default-on/tiny-live, parameter search, or promotion.
- `0604T009` business execution is complete and awaiting QA. It combines `0604T006/T008` signal ranking and `0604T007` horizon/regime diagnostics into decision-time-visible candidate/watch/reject regime definitions only. Current result: `candidate_for_milestone3_executability=1`, `watch_needs_more_samples=6`, `reject_unstable_direction=9`, `reject_concentrated_or_aliased=2`. The only candidate is anchored by `binance_mid_move_ticks_from_prev`; `watch_regime_dependent` features remain secondary filter/context only.

## Next Step

Current controller decision point after `0518T004` QA:

```text
0513T002 QA passed.
0513T003 QA passed.
0512T008 QA passed.
0513T004 QA passed.
0513T005 QA passed.
0513T006 QA passed.
0513T007 QA failed due to Binance snapshot bootstrap bug in the sidecar reconstructed book.
0513T008 QA passed. It collected one no-rule control run and did not enable new strategy rules, promote live, modify strategy behavior, modify canonical audit schema, or modify core/connector APIs.
0513T009 QA passed. It used the existing 5-13-day-control-30min sample only, fixed the T007 snapshot/bootstrap bug, and did not start live or change strategy/core/connector/schema behavior.
`0519T009` QA passed. The new sample verifies all 15 T006 quote-update audit fields and reruns T008, but Step 9 remains default-off offline replay only and cannot be promoted from a single sample.
`0519T008` QA passed. The runner / artifact mechanics are accepted, while the old `5-13-day-control-30min` result remains `needs_more_instrumentation` because it lacks T006 fields.
`0519T010` QA passed as planning-only Step 9C.
`0518T004` implemented only anchor arbitration, side-conservative rounding, clamp, post-clamp re-check, guarded fallback, stale/join-age suppression, and diagnostic counters. It did not widen into source-level drift repair, generic quote-control redesign, or live promotion.
```

Current formal task:

```text
0513T005 QA passed.
0513T006 generated latency / market-data integrity / provenance / sample usability artifacts over existing local samples only.
Classification:
- 5-13-day-control-15min: pricing_research_candidate, limited to compressed BBO/mid sanity.
- 5-11-night-active: compressed_action_path_only.
- 5-10-day-control-1h-06: compressed_action_path_only.
- 5-9-noon: compressed_action_path_only.
- 5-9-small: compressed_action_path_only.
No current sample qualifies as queue_fill_research_candidate.
Current task: `0525T001` is `待验收`. `0521T002` has passed QA. `0520T002` is complete. T008 runner is accepted, T009 supplied one current-format T006 sample, T010 defines the multi-sample validation contract, T011 collected 3 separated 30min current-format no-rule/default-off samples with run ids beginning `5-19-night-active`, and T001 validated the multi-sample set.
Step 9C plan direction:
- Treat data scenario coverage as the immediate blocker.
- Require current-format samples with T006 fields, maker acceptance, sidecar/join quality, Stage 5 labels, Step 5C diagnostics, and Step 9B outputs.
- Compare candidate stability across volatility, spread, trade intensity, stale/latency, API/churn, inventory, post-only safety, cancel-fill, and market-view quality regimes.
- `0520T001` business execution is complete: accepted-set meets Step 9C research-comparison mass, clean-only sensitivity is under threshold, no candidate is `ready_for_tiny_live_design`, and no live/default-on/promotion or runner change was made.
- `0520T002` completed the validator hardening step and passed QA. `0521T001` collected `5-21-day-control-60min` successfully and passed QA. `0521T002` extended the multi-sample set with that sample and still found no `ready_for_tiny_live_design` candidate.
Completed prerequisites:
- `0514T003` passed QA after implementing and running Stage 4 read-only pricing-model research.
- `0514T004` passed QA as the maker execution outcome requirements contract.
- `0514T005` passed QA after implementing the read-only execution outcome label runner and dataset validation on `5-13-day-control-30min`.
Completed prerequisites:
- `0514T006` QA passed. Stage 6A now fixes the comparison unit, common labels, strata, sample policy, and Stage 6B boundary.
- `0514T007` QA passed. It proves methodology on one sample and shows replay fill/cancel lifecycle mismatch remains the dominant blocker.
- `0514T008` QA passed. It fixes the next-step direction: diagnosis first, repair second, sample expansion later.
- `0515T001` QA passed. It provides read-only diagnosis tables and confirms the root issue is replay lifecycle semantics, not sample coverage.
- `0515T002` QA passed. It narrows the repair to cancel-requested fill eligibility, terminal-state semantics, and long-horizon persistence bias, and explicitly defers sample expansion until after repair regression.
- Prepared next task:
  - `0515T005` should be a narrow repair task limited to the `cancel_race_window_too_short` residual class identified by `0515T004`.
  - It should explicitly exclude `4948` / `residual_replay_fill_trigger_uncertain`.
  - After `0515T005`, `0515T006` should remain a single-case read-only diagnosis for `4948` rather than a broad new repair.
  - `0516T001` adds top5 visible queue and same-price trade-quantity evidence for `4948`; QA should decide whether this is enough to plan a future queue-proxy repair design task. It does not authorize implementation.
  - `0516T002` measured repeatability of queue-ahead proxy mismatch: proxy-only no-fill pattern repeats, but replay-fill false-positive remains single-case.
  - `0518T001` should convert this into a conservative gate design and explicitly defer implementation until more data / more replay false-positive cases exist.
```

Prepared next task:

```text
0511T001 - adverse-selection timing rule 设计规格与验收合同
Dispatch to 业务线程-python for design only. Do not modify strategy code in T001.

0511T002 - adverse-selection timing guard default-off 实现与 Stage 6J replay
Dispatch only after 0511T001 design contract is accepted.

0511T003 - 升级 workflow dashboard 为实验决策看板
Can run independently as a workflow display-layer improvement. Do not modify strategy code.

0511T004 - adverse timing trigger 未命中原因诊断
Dispatch to 测试线程 after 0511T002 outputs are available. Diagnose whether target_deterioration did not trigger, triggered off-path, or replay/source-path metrics are insensitive.

0512T001 - 对齐 Stage 6J replay 与 live adverse-selection source-path
Dispatch to 测试线程 after 0511T004 QA. Diagnose why live/current-format risk diagnostics show positive adverse-selection counts while Stage 6J replay summary shows zero.

0512T002 - add-side submit/re-add toxic timing rule 设计合同
Dispatch to 业务线程-python only after 0512T004 defines the replay/live observability gate and sample split. Design only; do not modify strategy code.

0512T003 - 5-11-night-active live 样本 replay/acceptance/cancel-fill 分析
Dispatch to 测试线程 after the 4H sample has been pulled locally. Complete audit replay, maker acceptance, cancel-fill risk analysis, and archive refresh.

0512T004 - Stage 6J / live adverse-selection 观测门禁改进合同
Dispatch to 测试线程 after 0512T001 QA. It must run before 0512T002 and define which evidence is action-path coverage, replay-model regression, or live-derived source-path proof. It must also define `5-11-night-active` as the main development/diagnostic sample and `5-10-day-control-1h-06` / `5-9-noon` / `5-9-small` as cross-sample sanity checks.

0512T005 - add-side toxic timing guard default-off 实现与离线 replay
Dispatch to 业务线程-python only after 0512T002 QA passes. Implement the default-off add-side toxic timing guard, audit fields, unit tests, Stage 6J replay, and action-path coverage reporting. Do not start live or default-enable the rule.

0512T006 - T005 blocked-row attribution 分析计划
Dispatch to 测试线程 after T005 QA, or earlier only if total controller explicitly allows a planning-only task while T005 is waiting for QA. Plan blocked-row stratification, actual submit-removal analysis, and reason attribution only. Do not implement, do not run experiments, do not design stricter candidates, and do not start live.
```

## Runner Status

- `0510T002` runner has been implemented and executed.
- Current `0510T002` status: `待验收`.
- Output directory: `local_live_analysis/stage6j_cross_sample_0510T002/`.
- Result: `diagnostic_only_no_promotion`, samples `4`, candidates `6`, hard failures `0`.
- Next action: QA should review `.workflow/reports/0510T002-business.md`.
- `0511T001` has been narrowed to a design-contract task only.
- `0511T002` has been created as the later implementation/replay task.
- `0511T003` has been created to upgrade the static dashboard from a task index into an experiment decision board.
- `0511T003` has been executed and is waiting for QA. The dashboard now shows decision summaries, key metrics, business results, QA conclusions, and next steps.
- `0511T001` has been executed and is waiting for QA. It recommends `0511T002` only as a default-off implementation/replay task, not as live promotion.
- `0511T002` has been executed and is waiting for QA. Result: `diagnostic_only_no_promotion`, samples `4`, candidates `7`, hard failures `0`; no live micro test allowed.
- `0511T002` is suitable for QA on implementation/replay mechanics, but it has not identified why the pure adverse timing strategy produced no incremental replay effect. Current known fact: the pure `adverse_timing_target_deterioration_*` candidates matched baseline; root cause is still open.
- `0511T004` has been created as a follow-up diagnostic task for the pure adverse timing candidates that matched baseline in `0511T002`.
- `0511T004` has been executed and is waiting for QA. Diagnosis: `target_deterioration` fired heavily, but it did not overlap `submit_buy` / `submit_sell`; baseline and pure adverse timing candidates had zero row-level action-path differences. Current Stage 6J replay also shows `guard_candidate_adverse_selection_count=0`, while live/current-format risk diagnostics show positive adverse-selection candidate counts. No live micro test.
- `0512T001` has been created as the next diagnostic task. It must align Stage 6J replay source-path observability before any further adverse timing implementation.
- `0512T002` has been updated as a later design-contract task for add-side submit/re-add toxic timing. It is blocked on `0512T004` and does not modify strategy code.
- `0512T001` has been executed and is waiting for QA. Diagnosis: Stage 6J replay does not replay live order lifecycle/cancel-to-fill races; it regenerates simulated order lifecycles through `run_backtest`. Current Stage 6J can remain a replay-model regression gate, but cannot alone prove live adverse-selection source-path improvement. `0512T002` may start as design-only and must include this limitation.
- `0512T003` has been executed and is waiting for QA. `5-11-night-active` passed maker acceptance with action/planned/reject/throttle all `1.0`, working semantic/blocking mismatch `0/0`, API/throttle mismatch `0`, strict replay lag breach/drop/fail `0/0/0`, and post-startup outside dual gate rows `0`. Cancel-fill risk repeated at 4H scale: `391/915` fills after cancel request, notional rate `0.427300`, add-side candidates `201`, adverse-selection candidates `190`. Cross-sample risk decision is `proceed_to_stage6j_narrow_rule`; no live micro test.
- `0512T004` has been created as the replay/live observability-gate task that must run before `0512T002`.
- `0512T004` and `0512T002` now explicitly use `5-11-night-active` as the main development/diagnostic sample and `5-10-day-control-1h-06` / `5-9-noon` / `5-9-small` as cross-sample sanity checks.
- `0512T001` QA passed. Stage 6J replay remains a replay-model regression gate only and cannot alone prove live adverse-selection source-path improvement.
- `0512T003` QA passed. `5-11-night-active` is accepted as the current-format main development/diagnostic sample; it does not authorize live micro test.
- `0512T004` has been executed and is waiting for QA. It defines the three-layer evidence contract: action-path coverage, replay-model regression, and live-derived source-path proof. It allows `0512T002` to start after QA only as a design-contract task; no implementation or live micro test is authorized.
- `0512T004` QA passed. It authorizes `0512T002` to start only as a design-contract task.
- `0512T002` has been executed and is waiting for QA. It defines a default-off add-side submit/re-add toxic timing guard contract, with shared helper, config, audit fields, Stage 6J replay matrix, action-path coverage requirements, and no-live boundary. It allows a later implementation/offline replay task after QA, but does not authorize implementation or live micro test.
- `0512T005` has been created as the post-`0512T002` implementation/offline replay task. It is `待执行` and blocked on `0512T002` QA. It does not authorize live micro test.
- `0512T002` QA passed. It authorized only the post-design implementation/offline replay task, not live micro test.
- `0512T005` QA passed. Commit `34c954e` implements the default-off `add_side_toxic_timing_guard`, shared live/backtest helper, audit fields, Stage 6J candidate matrix, action-path coverage reporting, and focused tests. Stage 6J result: `diagnostic_only_no_promotion`, samples `4`, candidates `7`, hard failures `0`; blocked reduce-side total `0`; pure toxic timing candidates show action-path coverage but no replay risk improvement, so no live micro test.
- `0512T006` has been created as a planning-only follow-up for the T005 phenomenon. It covers only blocked-row stratification, actual submit-removal analysis, and reason attribution plans. It explicitly excludes stricter candidate design, implementation, new replay runs, and live.
- `0512T006` has been executed and is waiting for QA. It produced a planning-only T007 attribution contract: blocked-row strata, true submit-removal risk linkage, and reason/window attribution. It did not implement scripts, run new replay, design stricter candidates, or start live.
- `0512T006` QA passed. It authorizes creation of T007 as read-only attribution / experiment implementation only; no stricter rule redesign or live micro test is authorized.
- `0512T007` QA passed. Result: representative 100ms pure toxic blocked rows `594`, true submit removals `56` row-level / `42` unique submit-order keys, blocked reduce-side `0`, removed-submit overlap with baseline cancel-fill risk events `0/56`, and 50/100/200ms blocked key equivalence `100%`. The direct explanation is that coverage mostly did not remove baseline submits, and the submits it did remove were not the replay risk orders.
- `0512T008` QA passed. Result: live/backtest both call `hbt.depth(0)`, but live reads connector-maintained depth while Stage 6J no-overlay reads replay-reconstructed depth. Audit replay overlay forces compressed market/fair/target fields but not top5 strings. Existing audit/npz is sufficient for compressed action-path alignment, but not sufficient for full L2 / queue / OFI / microprice equivalence.
- `0513T001` QA passed. It authorizes only a bounded `0513T002` implementation for strategy-layer MarketView provenance / top5 audit transparency. It does not authorize live, replay candidates, core API changes, converter/npz changes, or microprice/OFI/queue work.
- `0513T002` QA passed. It adds strategy-layer MarketView provenance and top5 source fields without changing strategy behavior, core API, configs, replay candidates, or live scripts.
- `0513T003` QA passed. It used git commit `192470f` to sync T002/T003 validation code to `awsserver1`, collected `5-13-day-control-15min` from `2026-05-13T08:35:45+0900` to `2026-05-13T08:50:58+0900`, ran `align_live_run.py`, ran `maker_acceptance.py`, and confirmed T002 provenance fields: live decision rows `live_depth`, normal replay decision rows `replay_depth`, and audit replay decision rows `market_view_source=audit_overlay` with `top5_source=replay_depth`. Acceptance hard gates passed, but top5/full L2 remain not fully aligned; no live promotion.
- `0513T004` QA passed. It added `examples/binance_tick_mm/deploy/preflight_live_run.py`, integrated it into `run_live.sh`, added focused tests, and verified startup preflight manifest generation. The gate records commit/dirty status/config hashes/key code hashes/schema compatibility/start-stop marker paths and fails before tmux/live if `AUDIT_FIELDS` is incompatible with strategy audit rows. No live run, AWS change, strategy semantic change, PnL proof, or full L2 proof.
- `0513T005` QA passed. It is planning-only for Step 2 and defines latency metrics, market-data integrity checks, sample priority, artifact outputs, sample usability classification, acceptance criteria, and follow-up tasks. It does not implement code, run replay, start live, or authorize pricing/queue research.
- `0513T006` QA passed. It generated the required Step 2 output artifacts under `local_live_analysis/step2_market_data_baseline_0513T006/`. Result: only `5-13-day-control-15min` is a limited `pricing_research_candidate`; the other four samples are `compressed_action_path_only`; no current sample supports queue/fill research. Existing samples remain legacy/pre-T004 and lack full raw provenance in converted npz.
- `0513T007` has been executed and is waiting for QA. It keeps the standard npz `data` main event array unchanged, adds Binance provenance/top5 sidecars, and proves `raw_seq -> final npz rows -> reconstructed top5 book -> decision rows` mapping with as-of join-age acceptance. Smoke metrics: final data row mapping coverage `1.0`, future join count `0`, depth `pu` mismatch `0`, bookTicker/depth BBO match/mismatch `151/1`; bounded slice also correctly reports unusable sync/join quality with `first_valid_update_aligned=false`, stale joins `243/244`, and gap-crossed joins `244/244`. It is explicitly top5-only and does not authorize live, replay/sweep, strategy changes, full L2 persistence, exact queue-position claims, `align_live_run.py` integration, canonical audit schema changes, or core/connector API changes.
- `0513T008` QA passed. It synced commit `f228950` through a clean remote git worktree, collected `5-13-day-control-30min` from `2026-05-13T17:30:12+0900` to `2026-05-13T18:00:25+0900`, passed T004 preflight with `dirty=false`, ran `align_live_run.py`, passed `maker_acceptance.py`, generated T007 full-run sidecar/join artifacts, refreshed archive sha256 `d7291afe0cb547663d4aa8e4cc9a175bfd06a3e7fcffea9d9eab82cc70b5c033`, and classified the sample as limited `pricing_research_candidate`. It is not live promotion and does not authorize strategy-rule changes.
- `0513T007` QA failed after T008 full-run validation exposed a sidecar bootstrap bug. On `5-13-day-control-30min`, snapshot `raw_seq=6` has `lastUpdateId=10537138804218`; buffered depth `raw_seq=5` has `U=10537138802913, u=10537138805036` and covers `lastUpdateId+1=10537138804219`; snapshot-after depth `raw_seq=7` has `pu=10537138805036` and should chain from `raw_seq=5`. Current T007 ignores the buffered update and starts at `raw_seq=7`, causing `first_valid_update_aligned=false` and `gap_crossed_join_count=47499/47499`.
- `0513T009` QA passed. It replays the buffered pre-snapshot `raw_seq=5` after snapshot `raw_seq=6`, then chains future `raw_seq=7` by `pu == previous_u`. Fixed full-run metrics on `5-13-day-control-30min`: first valid update aligned `true`, depth `pu` mismatch `0`, final data row mapping coverage `1.0`, decision join coverage `1.0`, future join count `0`, and gap-crossed join count `0`. The sample is upgraded for top5 microprice / OFI proxy / imbalance pricing research candidates, but still not for full L2 equivalence, exact live-audit-vs-sidecar top5 equality, or exact queue/fill proof.
- Step 2 is complete enough to proceed to Step 3. The remaining top5 tick/qty non-exact matches should be handled as market-view acceptance thresholds/classification, not more T009 bootstrap repair.
- `0514T001` QA passed. It adds optional Stage 3 market-view gates to `maker_acceptance.py`, keeps the existing action-path gates intact, and classifies `5-13-day-control-30min` as `passes_pricing_research_market_view` using T009 fixed sidecar/join metrics. It does not authorize full L2 equivalence, exact queue proof, strategy changes, or live promotion.
- Stage 4 preconditions are satisfied for read-only pricing-model research. The next task should study fair-value candidates and markouts over accepted samples, not implement a strategy rule or start live.
- `0514T002` QA passed. It defines candidates, filters, horizons, outputs, acceptance criteria, and authorizes `0514T003` as the implementation task for a read-only pricing research runner.
- `0514T003` QA passed. It generated read-only research artifacts under `local_live_analysis/5-13-day-control-30min/stage4_pricing_research_0514T003/`: `47499` decision rows, `47067` primary non-stale rows, `432` stale rows excluded from primary, and `14` candidate_for_followup signals. QA grouped the discovered signals by importance and documented duplicate signals.
- `0514T004` QA passed. It is a requirements-only follow-up for maker execution outcome research: fill probability, time-to-fill, adverse selection after fill, spread capture, queue/priority proxy, cancel-to-fill race, reject/throttle/churn, inventory impact, quote placement/distance, missed-fill opportunity cost, realized PnL decomposition, tail risk, partial-fill lifecycle, inventory cycle, and sample validity/censoring. It requires later analysis to use statistics appropriate to each label type rather than a single universal correlation metric.
- `0514T005` QA passed. It implements the read-only execution outcome label runner, focused tests, and dataset validation on `5-13-day-control-30min`. Result: the first execution-outcome label layer now exists, and it pushes Stage 6 toward replay/live fill-cancel lifecycle proxy calibration rather than broad exact-queue language.
- `0514T006` QA passed. Result: Stage 6 is now explicitly framed as replay/live fill-cancel lifecycle proxy calibration; comparison should be keyed on matched submit opportunities rather than raw cross-domain `order_id`; `5-13-day-control-30min` is sufficient for single-sample methodology but not enough alone for quote-adjustment promotion; `0514T007` remains a read-only implementation task.
