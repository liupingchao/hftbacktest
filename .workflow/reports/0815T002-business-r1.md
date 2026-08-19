# 线程回报

执行线程：
- 业务线程-python/research

任务ID：
- 0815T002

状态：
- 待验收

更新时间：
- 2026-08-15 20:27 CST（星期六）

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_trigger_parity_admission.py`
- `examples/hyperliquid/test_cross_exchange_trigger_parity_admission.py`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage03_detector_parity/`
- `.workflow/reports/0815T002-business-r1.md`
- `.workflow/tasks/0815T002.md`

action：
- 仅修复 admission/runtime/archive/package identity；未修改 shared trigger、
  historical builder、frozen trigger contract、candidate projection 或三
  session/segment parity 语义。
- 自包含 baseline：
  - build 仍对 external pre-extraction baseline 做 before/after 校验；
  - 正式 bindings 删除 external baseline root，`ephemeral_baseline_refs=0`；
  - verify-only 只验证 package 内 `pre_extraction_source/`、
    `fixture/baseline_inventory.json` 和 `fixture/pre|post/`；
  - 将 external baseline 真实原子移走后，source/archive verify-only
    均 `rc=0`，随后已恢复 baseline。
- Verifier identity：
  - package 内三个 runtime sources 与固定 worktree current source
    exact bytes 绑定；
  - package 内三个 current tests 归档到 `runtime_tests/`，并与固定
    worktree current test bytes 绑定；
  - 修改 archived admission 后同步重算 runtime SHA、artifacts 和 core，
    source 与被修改的 archived CLI 均 fail closed。
- Exact artifact allowlist：
  - 冻结 `23` 个 artifact relative paths；manifest 自身固定为
    `parity_manifest.json` 且明确不进入 artifacts/core list；
  - verify 在读取 manifest 前扫描完整 file/directory universe；
  - unknown、Aug07/0807、outcome、response、markout、PnL、private、
    account、order、cancel 和额外 fixture episode path 即使 coherent
    rehash 也 fail closed；
  - episode 路径唯一允许例外精确为
    `fixture/pre/episodes/segment_0001.csv.gz` 和
    `fixture/post/episodes/segment_0001.csv.gz`。
- Default zero-write：
  - sibling imports 前设置 `sys.dont_write_bytecode=True`；
  - 默认 `python`（无 `-B`）调用 source/archive verify-only 均成功，
    package path/bytes/SHA/mtime 不变，无 `.pyc`/`__pycache__`。
- Canonical manifest：
  - raw bytes 唯一要求为
    `json.dumps(indent=2, sort_keys=True) + newline`；
  - semantics-identical minified manifest 在 source/archive verifier
    中均 fail closed。
- 新 binding universe 为 `226` rows，before/after 各 `113`：
  - accepted dependency `2`
  - historical opaque inventory `3`
  - historical trigger truth `6`
  - detector input `79`
  - current source/test `6`
  - package internal baseline `11`
  - package internal source/test `6`

package identity：
- 正式 package：
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage03_detector_parity/`
- 旧 package identity 已作废：
  - core：
    `837f633ee0de74b3c223eb3491b11399d1445dce9dd9b6f563778d2847569fb7`
  - full：
    `959eccb87a834ab32c5e3d6ae1413b60616f8cc083a66738e8173e9e594a7518`
- 新 package：
  - core SHA256：
    `994c12f98dadb7201ad03b67d3c116caa85778233b9c1bc58fccc6b799124fb4`
  - full inventory SHA256：
    `7bff0caf92856fdda0115fabd5fdc066bee49ea4cfba7f2091a5ccc63b38386e`
  - manifest SHA256：
    `16c5a3fb2d0ee7503ccf0c4cba7890fbb15a8035445023f4a16ac6f2bf197260`
  - `24` files / `40,499,445` bytes / `23` manifest artifacts。
- Source/archive identities：
  - shared trigger：
    `69f6dd56b1e34a07f4073269cd3df11101cf630a78d72707af00dbfb2ef478ad`
  - historical builder：
    `7e8a2718b01d968d99843226a006ac451052a0ecd22dd086f458e49651452050`
  - parity admission：
    `efa18f6fdb4719e3833b3ee9799f30fcc0469cc3c8048ee5c794d5207f28e686`
  - trigger test：
    `e5128200a0e30a41c4cb7117ec27cb95bda773ea087fe19ae04e06e9eeb7bfcf`
  - builder test：
    `158c20cd012d32580fc37b7097a0caa7f3938e6a0fd354a5a0766eee0f80b64e`
  - admission test：
    `cd2a11e3fdc1cb0bcb72fc45c661cce269418b1704d14d0a3ae7bc70caa3ef34`

production parity：
- Frozen contract SHA 保持：
  `a894079e405073400d86f8471fd20e2a3a7116d3b804952bab8db56587f93b3d`。
- Candidate projection SHA 保持：
  `68e65a612d7bce2911a4e3c96fd10528914290ca045c8912e6be45045fd0415f`。
- `detector_parity_by_session.csv` SHA 保持：
  `67b8035aa5b8ce0cf650c6e8465aa83284e741f6d8ba3c92f492274c7267f534`。
- `detector_parity_by_segment.csv` SHA 保持：
  `c4ab6a2a23e6349c5a77a9f30373b09f1ffa3724d278afe49f24047444a736f9`。
- Jul30：`268522 / 141768 / 126754`，row-stream SHA
  `e2b4e998b87c81c1f8350a92bd1253db35fcb396cabe557773364535190ceed4`。
- Aug03：`127622 / 82533 / 45089`，row-stream SHA
  `73b8e5ae2ad6652c811e26c7df60d5cacba1a4febe812c0960450a7ea88fc309`。
- Aug04：`67468 / 43253 / 24215`，row-stream SHA
  `fa537846e50760a50ee38882c8d39884d6f240f3f844c3ae8b1b66fe10367dcf`。
- Aggregate：`463612` candidates / `267554` primary / `196058` rejected；
  `19` segments；36-field text、row order、candidate sequence mismatch 均
  为 `0`。
- Projection、contract、session/segment CSV、pre-extraction source 和
  全部 fixture pre/post bytes 与旧 package 逐 byte 相同。

determinism：
- Formal：
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage03_detector_parity/`
- Build A：
  `/tmp/0815T002-r1-parity-a.AQc2i0/package`
- Build B：
  `/tmp/0815T002-r1-parity-b.Wce87m/package`
- 三者均为 `24` files / `40,499,445` bytes，全部 relative paths、
  bytes、逐文件 SHA、core 和 full inventory byte-identical。

immutability：
- Accepted Stage 1 unchanged：
  `14` files / `1,401,387` bytes / core
  `9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96` /
  full `c540cc056313716b3bdd2b9c0fe076cda15a7f152b399ae6a1283b3aa8aa6590`。
- Accepted Stage 2 unchanged：
  `16` files / `19,834,728` bytes / core
  `7b3d06c3225f77c9929cdd3fa40d69c0866d83dbb75dd584dcb7db6db22ad3f8` /
  full `bdade16c53aed7bba54fdb9408ba3a8a03036d183720f589bda2a762448a3833`。
- Historical package full inventories unchanged：
  - Jul30：`11` files / `79,979,087` bytes /
    `603565e840ff99fdf050f9c862c382524934ef8e2034cf5bc668bfdf8d7a6387`
  - Aug03：`13` files / `46,747,139` bytes /
    `50a95175db0e0abc36604e1f6d3481d094270ee9ceebf24259356dfe8479e648`
  - Aug04：`4` files / `24,531,687` bytes /
    `1a182c9ae346e6a3c2989734d18316ffc78e26bc3f40aeadc335f1a625c856c3`
- Detector-only inventories remained at accepted frozen SHA/count/bytes in all
  three builds；未读取历史 episode response rows 或 Aug07 event rows。

verify：
- Stage 3 required suite：
  `43 passed in 117.26s`。
- Focused admission suite（含 6 类新增 production-size hostile
  regressions）：
  `17 passed in 115.84s`。
- Alignment/density：
  `183 passed in 8.74s`。
- Merging/recovery：
  `33 passed in 0.13s`。
- 新 hostile regressions 分别覆盖：
  self-contained baseline、archived verifier coherent rehash、exact
  artifact allowlist、default source/archive zero-write、current test
  archive/binding、canonical manifest raw bytes。
- Source/archive `--help` 均 `18` lines 且 bytes identical。
- Source/archive 默认 `python --verify-only` 均 `rc=0`、相同 core/count，
  package full path/bytes/SHA/mtime unchanged。
- Ruff：`All checks passed!`
- 外置 `PYTHONPYCACHEPREFIX` compileall：pass。
- Formal 全部 gzip：`gzip -t` pass。
- Formal/A/B：`diff -rq` 无差异。
- Scoped `git diff --check`：pass。

boundary：
- `aug07_event_rows_read=false`
- `episode_v3_built=false`
- `historical_episode_rows_read=false`
- `network_accessed=false`
- `outcome_or_model_computed=false`
- `private_or_order_endpoint_accessed=false`
- `response_rows_read=false`
- 未开始 Stage 4；未访问网络、AWS、SSH、private/account/order/cancel；
  未采集、未下单、未修改 live config。
- 未修改 `task_plan.md`、`progress.md`、`findings.md` 或
  `docs/qa-acceptance-report.md`。

done：
- 第一次独立 QA 的 `P1=3 / P2=2 / P3=1` bounded repair 已全部实现并
  通过业务线程 focused/full verification。
- 当前 task 状态回到 `待验收`；等待全新独立 QA，未自行派发 QA，
  未开始 Stage 4。

blockers：
- 无。

commit：
- 无

提交信息：
- 无
