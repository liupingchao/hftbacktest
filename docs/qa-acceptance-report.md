# QA 验收结果

## Findings（按严重性）

- P0/P1/P2：无。
- P3：Transfer manifest 记录 bundle SHA-256，但未提交 bundle bytes 或
  base-range receipt，无法离线单独重算。Exact source commit/blob、setup
  checkout、runner hash 和 collection/post-state identity 已覆盖 source
  identity，本项不阻断验收。

执行线程：
- QA验收线程

任务ID：
- 0722T066

状态：
- 已通过

更新时间：
- 2026-07-22 18:10 Asia/Shanghai

验收对象：
- source implementation
  `903a3e68284942852cf30997c6fd19c960995afc`
- evidence/workflow
  `0705aac9991e568ce9d4c718c25db1fc8d51f63d`
- readiness
  `50ec3fa3e6f73428fecd0856fb46c31827594d08`

实际结果：
- Source runner blob 与 recorded runner SHA-256
  `da8c30aae4e72001f182ceda3699c4480f8ffd5e2e564431b81ac50e25c72d4f`
  一致。
- Setup、collection 和最终 post-state receipts 的 command、instance、
  source identity 与成功状态匹配；第一次失败的 post-state attempt 已被新的
  successful command 替代。
- Remote manifest 12/12 pulled root file hashes 通过。
- 从 884 committed event rows 独立重建 280 exposures：buy/sell 各
  `140 observations / 4 distances`，fixed grid 与 directional
  at-or-through semantics 精确匹配。
- 独立 OLS 精确复现双侧 A、k、RMSE、confidence 和 confidence bounds。
- Seed exact 17 fields，自哈希与 expected hash 均为
  `e35c7fd8f3ec8268e5d50c7963889b73409470c9f01f92ca3a0d598a96562be9`；
  strict loader 与 hostile tamper fail closed。
- 8/8 root/offline core、两份 clean checkout 的 9-file rebuild 和 existing
  estimator `snapshot_match=true` 均通过。
- Terminal candidate 仍为
  `fallback_fixed / missing_latest_market_estimator`；T066 未宣称 live
  candidate、fill/economics、promotion 或 live authorization。
- Focused `30 passed in 0.23s`；full Hyperliquid
  `1297 passed, 2 skipped in 56.17s`；`py_compile` 和 `git diff --check`
  通过。
- QA 未访问 AWS、网络、credentials、private/account/order/cancel/service/live。

验收结论：
- 已通过。
- 结论说明：
  - Source-pinned public intensity seed、strict loader 和 deterministic
    evidence contract 可离线独立复现，且 overclaim boundary 正确。

通过项：
1. Source/receipt/manifest identity。
2. Independent exposure and regression rebuild。
3. Exact seed/hash/loader hostile contract。
4. Deterministic rebuild、replay 和 no-live boundary。

不通过项：
1. 无。

缺陷清单：
1. 无阻断缺陷；P3 bundle traceability 见详细 QA 报告。

阻塞项：
- T066 无。
- 后续真实订单仍需新的 formal task、current dynamic candidate `pass` 和
  fresh exact live authorization。

建议总控下一步：
1. 派发 exact seed wiring/pre-submit current-candidate gate 的独立任务。

提交信息：
- QA commit：由本报告提交后的线程回报提供。
