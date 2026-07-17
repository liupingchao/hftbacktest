# 线程回报

执行线程：
- 业务线程-live-safety-audit

任务ID：
- 0717T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0717T004.md`
- `.workflow/reports/0717T004-business.md`
- `.workflow/reports/0717T004-qa.md`
- `docs/qa-acceptance-report.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- Parsed `trade_logs/0717trade_history.csv` read-only and summarized all WTIOIL rows.
- Ran read-only SSM process/service/config/log audits on awsserver1.
- Inspected `xemm.service`, `/home/admin/XEMM_rust/config.json`, XEMM env key names only, and focused `journalctl` slices around the WTIOIL trade time.
- Did not run live, submit orders, cancel orders, stop services, restart services, or print credential values.

verify：
- 0717 trade history WTIOIL summary:
  - total WTIOIL rows: `251`
  - major pattern: long-running CL/WTI history from April onward, not a one-off 0717 event.
  - recent WTIOIL rows include repeated approximately `90 USDC` notional trades and 30-minute open/close cycles.
  - target row: `2026/7/17 13:41:40`, `WTIOIL (xyz)`, `Open Short`, `78.51`, `1.14`, notional `89.5014`.
- awsserver1 process/service audit:
  - running process: `/home/admin/XEMM_rust/target/release/xemm_rust`
  - service: `xemm.service - XEMM Rust Bot`
  - active since `2026-07-17 12:20:07 JST`
  - systemd unit uses `WorkingDirectory=/home/admin/XEMM_rust` and `EnvironmentFile=/home/admin/XEMM_rust/.env`.
- XEMM config redacted facts:
  - `maker_exchange`: `binance`
  - `maker_symbol`: `CLUSDT`
  - `hedge_symbol`: `xyz:CL`
  - `symbol`: `CL`
  - `order_notional_usd`: `90.0`
  - `profit_rate_bps`: `80.0`
  - `order_refresh_interval_secs`: `18`
- Focused XEMM journal evidence:
  - `2026-07-17T05:41:39.493870Z`: recovered Binance fill `BUY 1.140000 @ $78.470000`.
  - `2026-07-17T05:41:39.507464Z`: executing `SELL 1.14` on Hyperliquid.
  - `2026-07-17T05:41:39.529202Z`: fetching asset metadata with `dex=xyz`.
  - `2026-07-17T05:41:39.630198Z`: Hyperliquid market order `SELL 1.14 xyz:CL`.
  - `2026-07-17T05:41:40.295954Z`: hedge filled `1.14 @ $78.51`.
  - `2026-07-17T05:42:20.742519Z`: trade summary says `Hyperliquid: SELL 1.1400 xyz:CL @ $78.510000`.
- The XEMM hedge log exactly matches the trade-history WTIOIL row by instrument family (`xyz:CL` / WTIOIL), side (`SELL` / open short), size (`1.14`), price (`78.51`), and timestamp.

done：
- The WTIOIL `Open Short 1.14 @ 78.51` was caused by `xemm.service` hedging a Binance `CLUSDT` maker fill into Hyperliquid `xyz:CL`.
- It was not caused by the 0717T002 cross-exchange Python live runner.
- The apparent "short symbol mismatch" is therefore not a Python runner symbol-routing bug. It is a concurrent-live-service contamination / account-isolation bug: another live bot on the same awsserver1 and account was trading CL/WTI while the BTC evidence run was being audited.
- This also explains why WTI history has many prior rows from April/June/July and why the 0717 WTI row notional is about `90 USDC`, matching XEMM config.
- Future live tests should not be interpreted while `xemm.service` is running on the same account; either stop/isolate that service under a separate authorization, or use a clean account/subaccount and provenance guard.

blockers：
- `same_account_concurrent_xemm_service_contaminates_live_evidence`
- `hyperliquid_rest_fill_source_missing_or_incomplete_for_xemm_hedge`
- `live_tests_need_account_or_service_isolation_before_next_run`

commit：
- 无

提交信息：
- 无
