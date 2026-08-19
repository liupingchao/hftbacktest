# 业务线程回报

执行线程：
- 业务线程-python/research

任务ID：
- 0815T001

状态：
- 待验收

更新时间：
- 2026-08-15（星期六）

是否进行QA验收：
- 是

QA说明：
- 请由新的独立 QA 线程执行第六轮验收，不沿用业务或前五轮 QA 结论。

files：
- `examples/hyperliquid/cross_exchange_trigger_density_admission.py`
- `examples/hyperliquid/test_cross_exchange_trigger_density_admission.py`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage02_density/`
- `.workflow/tasks/0815T001.md`
- `.workflow/reports/0815T001-business-r5.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 修复 Round 5 QA P1：所有 `input_bindings.csv` 行在分组前先通过
  `snapshot_phase` 和 `binding_scope` 精确枚举闭包，并对所有行执行
  Aug07/outcome-like 路径检查。
- 唯一 Aug07 路径例外为 accepted Stage 1 包内已验收的控制证据
  `consumption_ledgers/aug07_access_ledger.json`；它证明
  `event_rows_opened=false`，不是 event-row 数据。例外同时要求精确
  scope、role、Stage 1 根目录和相对路径，其他 Aug07 路径仍失败关闭。
- 修复 Round 5 QA P2：`reports/trigger_density_admission.md` 由已验证的
  density、episode 和 ESS CSV 唯一生成，verify-only 逐字节比较规范
  报告，不能再通过重哈希发布错误计数。
- 修复 Round 5 QA P2：`density_manifest.json` 要求精确 top-level
  key-set 和规范 pretty-JSON bytes；额外键、缺失键或非规范序列化均
  失败关闭。
- 修复 Round 5 QA P3：小型摘要 CSV 的 producer/verifier 使用同一
  运算顺序和规范文本，逐字段精确比较，不再使用任何非零浮点容差。
- 上述 binding、manifest、report 和 exact-value 规则全部进入
  `frozen_density_contract.json` 的 `verification_closure`。
- 新增永久回归覆盖 unknown phase/scope + forbidden path、Stage 1
  Aug07 控制账本精确例外、错误报告数值、manifest extra key/非规范
  JSON 和 `5e-10` 小型摘要浮点漂移。
- 从冻结只读输入重新原子发布正式包，并完成两个不同 `/tmp` 目录的
  独立完整构建。
- 未更改 detector、candidate population、merging mapping、sensitivity、
  ESS 或任何研究结果。

verify：
- Admission suite：`91 passed in 2.17s`。
- Stage 2 四文件完整 focused regression：`142 passed in 8.61s`。
- Reused hierarchy recovery regressions：
  `3 passed, 19 deselected in 0.03s`。
- Ruff、compileall、CLI help 和 scoped `git diff --check` 全部通过。
- Source/archived `--help` 均为 `26` 行且逐字节一致。
- 正式包 source CLI、archived CLI `--verify-only` 均通过，报告：
  - `artifact_count=15`；
  - `candidate_membership_rows=463612`；
  - `confirmed_rows=267554`；
  - `sensitivity_membership_rows=463612`；
  - `session_count=3`；
  - core SHA256
    `f29597985d41e0f33588513a5a940469d0bb6188ff0f33acbdb50ca115a7a638`。
- Production-sized Round 5 修复证据位于
  `/tmp/0815T001-r5-negative.FPtc2R/`。四类副本均同步刷新 artifact
  records/core hash，source/archived verifier 共 `8/8` 次 `rc=2`：
  - unknown phase/scope + `/tmp/0807/outcome_rows.csv.gz`：
    `forbidden Aug07 input binding`；
  - Jul30 report count 改为 `999999`：
    `admission report canonical content drift`；
  - coverage fraction 增加 `5e-10`：
    `window_union_coverage_fraction drift`；
  - manifest 增加 `qa_status`：
    `density manifest key-set drift`。
- 正式包、Build A
  `/tmp/0815T001-r5-density-a.hwOreI/package` 和 Build B
  `/tmp/0815T001-r5-density-b.wu3uaX/package` 均为：
  - `16` files；
  - `19,831,761` bytes；
  - full-directory inventory SHA256
    `ea1aeee9aae8a38cdeb3eed0471b94c45248785e241f65a3e8e120f8dc2baa29`；
  - core package SHA256
    `f29597985d41e0f33588513a5a940469d0bb6188ff0f33acbdb50ca115a7a638`。
- Formal / Build A / Build B 的全部 relative paths、bytes 和逐文件
  SHA256 完全一致；source CLI 对 A/B 的 `--compare-to` 均
  `identical=true`。
- Archived verify-only 前后正式包 inventory 均为上述值；无
  `__pycache__`、`.pyc` 或新增路径。
- 三次构建 source inventory before/after 均为 `123` files、
  `169,677,903` bytes、SHA256
  `94bad85bfd4981b351f84c53628099468ec27f13d308402ac7125a9d582a6644`。
- 三次构建 accepted Stage 1 inventory before/after 均为 `14` files、
  `1,401,387` bytes、SHA256
  `c540cc056313716b3bdd2b9c0fe076cda15a7f152b399ae6a1283b3aa8aa6590`。
- Stage 1 core SHA256 保持
  `9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96`。
- 相对 Round 5 QA 对象只变化：
  `density_manifest.json`、`frozen_density_contract.json`、
  `runtime_source/cross_exchange_trigger_density_admission.py`。
- 正式 manifest boundary 七项继续全部为 `false`。

unchanged research evidence：
- Jul30：`268522 / 141768` candidates/confirmed，
  `18.646809 / 9.844708` 每秒；`39928` clusters、`10536` flows、
  `9` candidate overlap blocks、`248/240` all/complete 60s blocks；
  Bartlett ESS `1096.308299 / 837.102917`。
- Aug03：`127622 / 82533`，`7.089796 / 4.584963` 每秒；
  `47193` clusters、`23397` flows、`232` candidate overlap blocks、
  `310/300` 60s blocks；Bartlett ESS
  `2195.991765 / 2085.852134`。该 session 仍
  `formal_eligible=false`。
- Aug04：`67468 / 43253`，`9.370386 / 6.007252` 每秒；
  `23113` clusters、`9451` flows、`6` candidate overlap blocks、
  `121/120` 60s blocks；Bartlett ESS
  `275.582545 / 263.249320`。
- Family A 2000ms window coverage 分别为 Jul30 `0.999823`、
  Aug03 `0.992360`、Aug04 `0.999856`，继续表明 trigger 是近连续过程，
  不能把 `463612` 行解释为 IID 支撑。

done：
- Round 5 的 P1/P2/P2/P3 已在同一 `0815T001` 范围内 bounded repair。
- 修复版正式包已发布到
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage02_density/`。
- 当前任务重新进入 `待验收`。Round 6 独立 QA `已通过` 前，Stage 3
  detector parity 及所有后续阶段继续锁定。

blockers：
- 无

commit：
- 无

提交信息：
- 无
