# 线程回报

执行线程：
- 业务线程-live-safety-audit

任务ID：
- 0717T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0717T003.md`
- `.workflow/reports/0717T003-business.md`
- `.workflow/reports/0717T003-qa.md`
- `docs/qa-acceptance-report.md`
- `task_plan.md`
- `progress.md`
- `findings.md`
- `trade_logs/0717trade_history.csv` read-only
- `local_live_analysis/cross_exchange_controlled_role_evidence_0717T002_20260717T045820Z/` read-only
- `examples/hyperliquid/` read-only

action：
- Parsed `trade_logs/0717trade_history.csv` using its actual lowercase headers: `time, coin, dir, px, sz, ntl, fee, closedPnl`.
- Reconciled 0717 CSV rows against the 0717T002 live window times and `order_intent_audit.csv`.
- Inspected 0717T002 artifacts: `order_intent_audit.csv`, `private_order_response_audit.json`, `user_fills_pullback_audit.json`, `live_fill_ledger.csv`, `fill_liquidity_role_evidence.csv`, `window_status.json`, `inline_reprice_attempt_matrix.csv`, and lifecycle/coverage files.
- Audited symbol routing code in `examples/hyperliquid/hyperliquid_tiny_live_real_order_executor.py`, `examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py`, and `examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py`.
- Ran a read-only SSM account-scope audit on `awsserver1`; command id `7b8e64ff-771d-4013-955c-ae990ad9a6a9`.
- Did not run live, did not submit orders, and did not cancel orders.

verify：
- `trade_logs/0717trade_history.csv`: parsed `293` rows.
- 0717 CSV coin counts: `WTIOIL (xyz)=251`, `BTC=29`, `HYPE/USDC=11`, `COMP=2`.
- 0717 CSV rows on `2026/7/17`:
  - `2026/7/17 00:55:26`, `WTIOIL (xyz)`, `Open Short`, `78.429`, `1.14`
  - `2026/7/17 01:25:26`, `WTIOIL (xyz)`, `Close Short`, `78.503`, `1.14`
  - `2026/7/17 13:11:50`, `BTC`, `Open Long`, `63422`, `0.00067`
  - `2026/7/17 13:36:03`, `BTC`, `Open Long`, `63150`, `0.005`
  - `2026/7/17 13:41:40`, `WTIOIL (xyz)`, `Open Short`, `78.51`, `1.14`
- If the CSV timestamps are browser/local Shanghai time, the two BTC rows map to:
  - `2026-07-17T05:11:50Z`, window 01, exact match to `BTC buy 0.00067 @ 63422.0`
  - `2026-07-17T05:36:03Z`, window 02, exact match to `BTC buy 0.005 @ 63150.0`
- 0717T002 submitted attempts:
  - window 01: `2026-07-17T05:11:49.420Z`, `BTC buy 0.00067 @ 63422.0`, status `resting`
  - window 02: `2026-07-17T05:36:00.974Z`, `BTC buy 0.005 @ 63150.0`, status `resting`
  - window 03: `BTC buy 0.005 @ 62874.0`, post-only reject, `asset=0`
  - window 03: `BTC buy 0.00304 @ 62882.0`, post-only reject, `asset=0`
- Hyperliquid meta read-only audit showed `asset=0` is `BTC`, not WTIOIL.
- SSM account-scope audit showed the awsserver1 env account and wallet-from-private-key are the same redacted address/hash, but for `2026-07-17T04:55:00Z` to `2026-07-17T06:05:00Z`:
  - `user_fills_by_time(..., aggregate_by_time=False)` count `0`
  - `user_fills_by_time(..., aggregate_by_time=True)` count `0`
  - recent `user_fills` count `0`
  - open orders count `0`
  - positions `[]`
- Code inspection found no production WTIOIL constant or branch in the Hyperliquid live runner path.
- Code inspection found symbol guards:
  - `SYMBOL = "BTC"`
  - `validate_order_intent()` rejects non-BTC intents.
  - `SDKHyperliquidClient.order()` sends `intent.symbol` to `exchange.order(...)`.
  - `fill_matches_intent_without_oid()` rejects fills whose `coin/symbol` does not match the intent symbol.

done：
- The 0717 CSV file must be treated as real counter-evidence to the previous 0717T002 "no fill" claim.
- The two 0717 BTC long rows are highly consistent with the 0717T002 live runner intents by timestamp, symbol, side, price, and size.
- The WTIOIL short row at `2026/7/17 13:41:40` falls inside 0717T002 window 03 if interpreted as Shanghai time, but it does not match any 0717T002 order intent. Window 03 submitted only BTC buy intents, and both were post-only rejects.
- There is no local code-path evidence that this repo runner can submit WTIOIL orders under the 0717T002 configuration.
- There is a serious unresolved provenance mismatch: the downloaded trade history contains fills that align with the runner, but the awsserver1 env account read-only API view reports no fills, no recent fills, no positions, and no user volume.
- Future live tests should remain paused until account provenance is closed with a first-class guard: every live artifact must record a redacted/hash account id, the exchange account/vault used for order submission, and the same id used for fill pullback and post-run account state.

blockers：
- `0717_trade_history_vs_awsserver1_account_scope_mismatch`
- `0717T002_no_fill_conclusion_invalidated_by_external_trade_history`
- `WTIOIL_short_not_attributed_but_unexplained_within_window`

commit：
- 无

提交信息：
- 无
