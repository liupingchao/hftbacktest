# 业务线程报告

任务ID：
- `0815T003`

标题：
- SKHYNIX-JUL30-FAMILY-A-B-EPISODE-V3-BUILD

状态：
- 待验收

执行线程：
- 业务线程-python/research

日期：
- 2026-08-15

提交信息：
- commit：无

## 执行结论

- 已完成 Jul30-only linked Family A/B Episode v3 builder、standalone
  admission CLI、focused tests、正式 package 和两个隔离完整构建。
- 未发现真实未解决 defect 或 blocker。
- Family A 精确 `268522`，其中 rejected/unconfirmed `126754`；每个
  candidate 恰好一个 anchor、Family A view 和 public-market
  outcome-or-censor row，所有 rejected `t_confirm_ns` 保持 null。
- Family B 精确 `141768`，与同一 candidate IDs linked，且仅覆盖
  accepted Stage 3 `primary_episode=true` candidates。
- Stage 4 状态只置为 `待验收`；本线程未启动 QA，也未启动 Stage 5。

## 正式产物身份

- package：
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3/`
- schema：`episode_v3_jul30_v1`
- contract version：`skhynix_jul30_episode_v3_contract_v1`
- contract SHA256：
  `2239a170e670b8af7bca901524101b0beb3abc834deadfff714137f1b83d7347`
- package core SHA256：
  `14d945253ecae82cdb73bdc365ae01ae6f614b8ed7fc9ea56e08d65c4ba77398`
- full inventory SHA256：
  `25c02766e4eb263d4ba4bf20cfe48b37a9812f20d7eb402ddbcffb7ffe450785`
- manifest SHA256：
  `c94150a71520db7db1f1bba36f9de5716514d1e3cff4cf12299bebe271d08266`
- artifacts/files/bytes：`106` / `107` / `1562104138`
- full relative-path/file-SHA list digest：
  `f6c15624183ff17ffe88967e4d2e2b0366c06c2a495c8229f6643d384ef882ab`

## Runtime Binding

- builder：
  `7f4b46b9494be4a101a8f774719870e715c97c57ba7c5cd3fc9b875b8701baf3`
- standalone admission：
  `73d63fee5ce1ec699627644c8f5e9de586ce61f8060f047a1404381d6f3b0713`
- builder focused tests：
  `4f9b2c6021860be2e607a55edbb5de5db0eb45793ec9f49cceaecbc7c01b060b`
- admission focused tests：
  `50e1ebc27bd0d1897e65d6177683669c694a5a6d00f8d08ff2d8f176ac20425d`
- source/archive builder `--help` 均为 `18` 行且 bytes identical。
- package 内 `.pyc` / `__pycache__`：`0`。

## Cardinality

- anchors / Family A / outcomes：`268522` / `268522` / `268522`
- Family B / confirmed：`141768` / `141768`
- rejected Family A：`126754`
- sparse range A/B：`268522` / `141768`
- fixed-grid A/B：`4564874` / `2410056`
- event-count A/B：`805566` / `425304`
- feature-ledger A/B：`23092892` / `15310944`
- unavailable feature rows：`2863855`
- clusters / flows / 2000ms overlap blocks：
  `39928` / `10536` / `9`
- source-event catalog rows / quality interval rows：`40` / `10`

| segment | A | B | rejected | grid A/B | event-count A/B | ledger A/B | outcomes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| segment_0001 | 48683 | 22104 | 26579 | 827611/375768 | 146049/66312 | 4186738/2387232 | 48683 |
| segment_0002 | 33988 | 18168 | 15820 | 577796/308856 | 101964/54504 | 2922968/1962144 | 33988 |
| segment_0003 | 40503 | 20548 | 19955 | 688551/349316 | 121509/61644 | 3483258/2219184 | 40503 |
| segment_0004 | 38392 | 20243 | 18149 | 652664/344131 | 115176/60729 | 3301712/2186244 | 38392 |
| segment_0005 | 32485 | 17627 | 14858 | 552245/299659 | 97455/52881 | 2793710/1903716 | 32485 |
| segment_0006 | 26895 | 15232 | 11663 | 457215/258944 | 80685/45696 | 2312970/1645056 | 26895 |
| segment_0007 | 24627 | 14260 | 10367 | 418659/242420 | 73881/42780 | 2117922/1540080 | 24627 |
| segment_0008 | 22949 | 13586 | 9363 | 390133/230962 | 68847/40758 | 1973614/1467288 | 22949 |

## Outcome And Censor Evidence

- interval-censored first-event outcomes：`627406`
- right-censored：`177826`
- segment-censored：`316`
- quality-censored：`18`
- epoch-censored：`0`
- point-coerced interval：`0`
- auxiliary-degraded grid rows：`4316`
- core-degraded grid rows：`0`
- segment_0002 的 `asset_context` / `main_all_mids` 退化只进入
  auxiliary mask；未全局 censor core BBO/fast-L2/trade outcomes。

## Invariant And Mismatch Counts

- missing/duplicate/extra candidate IDs：`0`
- synthetic rejected confirm：`0`
- future feature observation mismatch：`0`
- anchor ordering mismatch：`0`
- cross-segment path mismatch：`0`
- cross-epoch path mismatch：`0`
- point interval coercion：`0`
- silently zeroed unavailable outcome：`0`
- 所有非空 Family A/B decision features 均携带
  `observed_at_ns/source_event_id/source_book_version/calculation_version`
  且满足 `observed_at_ns <= landmark`。

## Shared Event Store

- Episode 内不复制重叠 source events；package 发布 accepted Jul30
  structured R0/timeline 的 canonical catalog 和 deterministic
  membership/range indexes。
- accepted source counts：
  - Binance hot：`10744733`
  - Hyperliquid hot：`541122`
  - Hyperliquid auxiliary：`49169`
  - common L2 timeline：`556861`
- source manifest SHA256：
  `c46c735d7933587af6eece4a9dd1bce241b3c093866efce976b1c3f952e72ce0`

## Dependency And Immutability

- Stage 1 core/full：
  - `9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96`
  - `c540cc056313716b3bdd2b9c0fe076cda15a7f152b399ae6a1283b3aa8aa6590`
- Stage 2 core/full：
  - `7b3d06c3225f77c9929cdd3fa40d69c0866d83dbb75dd584dcb7db6db22ad3f8`
  - `bdade16c53aed7bba54fdb9408ba3a8a03036d183720f589bda2a762448a3833`
- Stage 3 core/full：
  - `4939d1c1addce493edb2f368297d56b37edd0b123de01497dcdee6e77637eb9b`
  - `ff8e3434672226371051151cea838503877dca79ac7860cf179256362d75e404`
- Stage 3 accepted QA report/mirror：
  `96d379061b2dbae0001e56050260fd60f38394fca1ab4ccdf60b1dd2f243b4d9`
- accepted Stage 1/2/3、Jul30 raw/R0/R1 和固定 Stage4 source/tests
  合并 input inventory before/after：
  `7ad44bf64b98e8e55e03f24458ce7702de038aa55184ab4e848cae9bedd3b78f`
  / `7ad44bf64b98e8e55e03f24458ce7702de038aa55184ab4e848cae9bedd3b78f`
- `input_inventory_unchanged=true`。

## Determinism And Zero-Write

- Formal、isolated Build A、isolated Build B 均独立完成 8 segments 和
  full admission。
- 三者 core/full/manifest SHA、107 个 relative paths、raw bytes 和
  per-file SHA 全部一致；三方 `diff -qr` 无输出。
- package 内 archived standalone admission `--verify-only` 通过：
  `verified=true`、A/B/rejected 为 `268522/141768/126754`。
- verify-only 前后逐路径 path/type/bytes/mtime/ctime 快照 SHA 均为
  `b72c38d14d49523ab14bad129c7a39a1169e036e148f3a6bb567ca44654d335c`，
  diff 为空。
- 正式 publication 使用 staging + fsync + atomic rename/exchange；失败
  staging 已清理，正式 package 目录外无 Stage4 partial publication。

## Tests And Static Checks

- Stage 4 focused：`12 passed in 0.08s`
- Stage 1/2 alignment+density regressions：`183 passed in 9.37s`
- Stage 3 detector/parity regressions：`49 passed in 122.17s`
- merging/recovery regressions：`33 passed in 0.23s`
- Ruff：`All checks passed!`
- 外置 pycache compileall：通过，`48` 个 bytecode files 全部写入
  `/tmp/0815T003-compile-pycache`。
- scoped `git diff --check`：通过。
- deterministic gzip header 在 full admission 中逐文件验证通过。

## Hostile Evidence

- production-size APFS clones 上完成 coherent artifact/core rehash 后，
  以下攻击均被真实 verifier fail closed：
  - confirmed-only Family A
  - 删除 rejected row
  - synthetic rejected confirm timestamp
  - confirmation feature 注入 Family A
  - feature `observed_at` after landmark
  - interval point coercion
  - cross-boundary sparse range
  - forward-fill counted as event
  - direction/risk-gap swap
  - unavailable feature zero-fill
  - unavailable outcome zero-fill
  - forbidden own-fill field
  - noncanonical CSV integer
  - dependency claim drift
  - source event-store catalog drift
  - unknown artifact
  - unknown empty directory
  - Aug03/Aug04/Aug07 event-path artifacts
  - noncanonical manifest JSON
  - archived verifier coherent rehash
- source-age deletion 触发 validator `ValueError`，没有被接受；正式 CLI
  的冻结异常边界会将 `ValueError` 转为 `rc=2`。
- degraded-evidence deletion 的首个 harness 使用了错误列名，修正后的
  专项复跑被用户中断；正式 verifier 在 segment output 前对
  `quality_intervals.csv` 与 accepted structured intervals 做 exact
  equality，并要求 aggregate `auxiliary_degraded_grid_rows > 0`。此项
  作为独立 QA 的单点复核，不构成已发现业务 defect 或 blocker。

## Hard Boundary

- `jul30_legacy_episode_rows_read=false`
- `aug03_aug04_future_event_rows_read=false`
- `aug07_event_rows_read=false`
- `model_or_score_run=false`
- `case_retrieval_run=false`
- `actionability_run=false`
- `network_accessed=false`
- `private_or_order_endpoint_accessed=false`
- `new_collection=false`
- `own_order_fill_pnl_fields=false`
- 未修改 accepted Stage 1/2/3 packages、历史 v1/v2 artifacts、
  Jul30 raw/R0/R1、`task_plan.md`、`progress.md`、`findings.md` 或
  `docs/qa-acceptance-report.md`。

## 待独立 QA

- 业务线程结论：待验收。
- 建议 QA 重点独立复跑 degraded-evidence deletion、default Python
  bytecode/zero-write 和 production-size partial-publication injection，
  并从冻结 Jul30 inputs 独立重建 package。
- 本报告不宣告 `已通过`，不派发 QA，不解锁或启动 Stage 5。
