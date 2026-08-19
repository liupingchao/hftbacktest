# 线程回报

执行线程：
- 业务线程-python/research

任务ID：
- 0815T002

状态：
- 待验收

更新时间：
- 2026-08-15 21:07 CST（星期六）

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_trigger_parity_admission.py`
- `examples/hyperliquid/test_cross_exchange_trigger_parity_admission.py`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage03_detector_parity/`
- `.workflow/reports/0815T002-business-r2.md`
- `.workflow/tasks/0815T002.md`

action：
- 本轮只修改两个 admission surfaces；未修改 shared detector、
  historical builder、projection、trigger contract 语义、session/segment
  CSV 或 Stage 4 内容。
- Dependency identity closure：
  - `verify_package()` 将 manifest 的
    `stage1_core_sha256`、`stage1_full_inventory_sha256`、
    `stage2_core_sha256`、`stage2_full_inventory_sha256` 与 hard-coded
    accepted constants 做 exact fail-fast compare；
  - `_verify_dependency_packages()` 从 fixed Stage 1/2 roots 实际读取
    manifest core，并重新计算 full inventory，不再用常量回填返回值；
  - `_verify_bindings()` 返回实际 verified dependency identities；
  - `verify_package()` 最后再次断言
    `manifest claims == accepted constants == actual fixed-root identities`。
- Canonical contract closure：
  - 将 canonical pretty JSON reader 泛化为带 label 的 raw-byte reader；
  - manifest 和 `frozen_trigger_contract.json` 均要求唯一
    `json.dumps(indent=2, sort_keys=True) + newline` bytes；
  - contract 先通过 raw-byte canonical check，再比较语义和 SHA。

hostile evidence：
- 新增 production-size dependency claim regressions：
  - `stage1_core_sha256` 单字段置零；
  - `stage1_full_inventory_sha256` 单字段置零；
  - `stage2_core_sha256` 单字段置零；
  - `stage2_full_inventory_sha256` 单字段置零；
  - 四字段组合置零。
- 每个 mutation 都保持 artifacts 和 accepted package core 不变；
  source API、source CLI、archived CLI 均 fail closed，错误为 dependency
  identity drift。
- 新增 production-size canonical contract coherent-rehash regression：
  - contract 改为语义相同 minified JSON；
  - 同步更新 contract SHA、artifact records、core 和 canonical manifest；
  - source API、source CLI、archived CLI 均以
    `frozen trigger contract canonical bytes drift` fail closed。
- 首轮六项 bounded repair regressions 全部复跑且无回归：
  self-contained baseline、archived verifier identity、exact artifact
  allowlist、default zero-write、current tests archive/binding、canonical
  manifest。
- 额外将 external baseline 真实移走后，source/archive 默认 verify-only
  仍均 `rc=0`；测试结束后 baseline 已恢复。

package identity：
- Round 1 repair identity 已作废：
  - core：
    `994c12f98dadb7201ad03b67d3c116caa85778233b9c1bc58fccc6b799124fb4`
  - full：
    `7bff0caf92856fdda0115fabd5fdc066bee49ea4cfba7f2091a5ccc63b38386e`
  - manifest：
    `16c5a3fb2d0ee7503ccf0c4cba7890fbb15a8035445023f4a16ac6f2bf197260`
- 新正式 package：
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage03_detector_parity/`
  - core SHA256：
    `4939d1c1addce493edb2f368297d56b37edd0b123de01497dcdee6e77637eb9b`
  - full inventory SHA256：
    `ff8e3434672226371051151cea838503877dca79ac7860cf179256362d75e404`
  - manifest SHA256：
    `a73d4a15a6f58bd5c533944131a9cadae4e74d1c6f0c23a65d75c35973282fa1`
  - `24` files / `40,504,469` bytes / `23` manifest artifacts。
- Updated source/test archive identities：
  - admission source：
    `c698337a4935c2ab089c69b526ddc2374eabefbb02f2b1c1aa127829c8253b34`
  - admission test：
    `1f02406de893f1a52da1aa707cd9bd535862aade2d9a6958c6c298a68f85731d`
  - input bindings：
    `b29a21e57429204576717ca4da3c38a9307e81d216efca74643ff9be006bee83`
- Unchanged source identities：
  - shared trigger：
    `69f6dd56b1e34a07f4073269cd3df11101cf630a78d72707af00dbfb2ef478ad`
  - historical builder：
    `7e8a2718b01d968d99843226a006ac451052a0ecd22dd086f458e49651452050`
  - trigger test：
    `e5128200a0e30a41c4cb7117ec27cb95bda773ea087fe19ae04e06e9eeb7bfcf`
  - builder test：
    `158c20cd012d32580fc37b7097a0caa7f3938e6a0fd354a5a0766eee0f80b64e`

determinism：
- Formal：
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage03_detector_parity/`
- Build A：
  `/tmp/0815T002-r2-parity-a.BDSPQj/package`
- Build B：
  `/tmp/0815T002-r2-parity-b.rSpQVs/package`
- 三者均为 `24` files / `40,504,469` bytes，全部 relative paths、
  raw bytes、逐文件 SHA、core、manifest SHA 和 full inventory
  byte-identical；`diff -rq` 无差异。

production parity：
- Candidate projection bytes/SHA 保持：
  `68e65a612d7bce2911a4e3c96fd10528914290ca045c8912e6be45045fd0415f`。
- Trigger contract bytes/SHA 保持：
  `a894079e405073400d86f8471fd20e2a3a7116d3b804952bab8db56587f93b3d`。
- Session CSV SHA 保持：
  `67b8035aa5b8ce0cf650c6e8465aa83284e741f6d8ba3c92f492274c7267f534`。
- Segment CSV SHA 保持：
  `c4ab6a2a23e6349c5a77a9f30373b09f1ffa3724d278afe49f24047444a736f9`。
- Jul30：`268522 / 141768 / 126754`，row-stream SHA
  `e2b4e998b87c81c1f8350a92bd1253db35fcb396cabe557773364535190ceed4`。
- Aug03：`127622 / 82533 / 45089`，row-stream SHA
  `73b8e5ae2ad6652c811e26c7df60d5cacba1a4febe812c0960450a7ea88fc309`。
- Aug04：`67468 / 43253 / 24215`，row-stream SHA
  `fa537846e50760a50ee38882c8d39884d6f240f3f844c3ae8b1b66fe10367dcf`。
- Aggregate：`463612` candidates / `267554` primary / `196058` rejected；
  `19` segments；field/order/sequence mismatch 全部为 `0`。
- Projection、contract、session/segment CSV 和全部 fixture pre/post files
  与 Round 1 package 逐 byte 相同。

dependency and immutability：
- Manifest claims、hard-coded constants 和 actual fixed-root identities
  三方精确一致：
  - Stage 1 core：
    `9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96`
  - Stage 1 full：
    `c540cc056313716b3bdd2b9c0fe076cda15a7f152b399ae6a1283b3aa8aa6590`
  - Stage 2 core：
    `7b3d06c3225f77c9929cdd3fa40d69c0866d83dbb75dd584dcb7db6db22ad3f8`
  - Stage 2 full：
    `bdade16c53aed7bba54fdb9408ba3a8a03036d183720f589bda2a762448a3833`
- Input bindings 保持 `226` rows，before/after 各 `113`，scope counts
  不变。
- Historical package full inventories unchanged：
  - Jul30：`11` files / `79,979,087` bytes /
    `603565e840ff99fdf050f9c862c382524934ef8e2034cf5bc668bfdf8d7a6387`
  - Aug03：`13` files / `46,747,139` bytes /
    `50a95175db0e0abc36604e1f6d3481d094270ee9ceebf24259356dfe8479e648`
  - Aug04：`4` files / `24,531,687` bytes /
    `1a182c9ae346e6a3c2989734d18316ffc78e26bc3f40aeadc335f1a625c856c3`
- Detector/pre-state inventories 保持 accepted count/bytes/SHA；
  source/test archives 与 fixed worktree bytes exact。

verify：
- Expanded focused admission suite：
  `23 passed in 116.96s`。
- Full Stage 3 suite：
  `49 passed in 117.42s`。
- Alignment/density：
  `183 passed in 8.77s`。
- Merging/recovery：
  `33 passed in 0.15s`。
- Source/archive `--help`：均 `18` lines 且 bytes identical。
- Source/archive 默认 `python --verify-only`：均 `rc=0`、相同 core/count；
  package root、directories 和 files 的 path/type/bytes/SHA/mtime_ns/
  ctime_ns 全部不变，无 `.pyc`/`__pycache__`。
- Ruff：`All checks passed!`
- 外置 `PYTHONPYCACHEPREFIX` compileall：pass。
- Formal 全部 gzip：`gzip -t` pass。
- Scoped `git diff --check`：pass。
- Formal/A/B exact identity：pass。

boundary：
- `aug07_event_rows_read=false`
- `episode_v3_built=false`
- `historical_episode_rows_read=false`
- `network_accessed=false`
- `outcome_or_model_computed=false`
- `private_or_order_endpoint_accessed=false`
- `response_rows_read=false`
- 未读取历史 response/outcome rows 或 Aug07 event rows；未开始 Stage 4；
  未访问网络、AWS、SSH、private/account/order/cancel；未采集、未下单、
  未修改 live config。
- 未修改 `task_plan.md`、`progress.md`、`findings.md` 或
  `docs/qa-acceptance-report.md`。

done：
- 第二次独立 QA 的 `P1=1 / P3=1` bounded repair 已全部实现；
  Round 1 六项修复无回归。
- Task 状态回到 `待验收`；等待第三轮全新独立 QA，未自行派发 QA，
  未开始 Stage 4。

blockers：
- 无。

commit：
- 无

提交信息：
- 无
