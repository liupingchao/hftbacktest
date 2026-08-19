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
- 请由新的独立 QA 线程执行第七轮验收，不沿用业务或前六轮 QA 结论。

files：
- `examples/hyperliquid/cross_exchange_trigger_density_admission.py`
- `examples/hyperliquid/test_cross_exchange_trigger_density_admission.py`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage02_density/`
- `.workflow/tasks/0815T001.md`
- `.workflow/reports/0815T001-business-r6.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 修复 Round 6 QA P1：新增 `EXPECTED_STAGE1_ROOT`，精确冻结 accepted
  Stage 1 resolved root：
  `/Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research/
  local_live_analysis/skhynix_trigger_aligned_episode_research_v1`。
- Builder 在读取任何 Stage 1 内容前要求 `stage1_dir.resolve()` 精确等于
  frozen root；替换、搬迁或不存在根目录均失败关闭。
- Verify-only 要求 `input_bindings.csv` 每一行的
  `stage1_package_path` 使用上述规范绝对文本；每条 Stage 1 scope path
  必须是该 root 下的规范绝对路径，不接受 `..`、其他 root 或 root 外
  路径。
- Verify-only 重新调用 accepted Stage 1 package verifier，实际读取并
  验证该真实目录的存在性、manifest artifact closure、core SHA 和完整
  `14` 文件库存；binding before/after inventory 还必须精确等于实时
  full inventory。
- Exact root、canonical absolute paths 和 actual package revalidation
  均进入 canonical frozen contract 的 `accepted_stage1` 与
  `verification_closure`。
- 新增永久回归：
  - coherently 替换全表 Stage 1 root；
  - builder 使用非 canonical Stage 1 root；
  - 构建后真实 Stage 1 文件 bytes 漂移。
- 从相同冻结只读输入重新原子发布正式包，并完成两个不同 `/tmp`
  目录的独立完整构建。
- 未更改 detector、candidate population、merging mapping、sensitivity、
  ESS 或任何研究数值。

verify：
- Admission suite：`94 passed in 2.64s`。
- Stage 2 四文件完整 focused regression：`145 passed in 8.81s`。
- Reused hierarchy recovery regressions：
  `3 passed, 19 deselected in 0.05s`。
- Ruff、外置 pycache compileall、CLI help 和 scoped
  `git diff --check` 全部通过。
- Source/archived `--help` 均为 `26` 行且逐字节一致。
- Production-sized Round 6 修复证据位于
  `/tmp/0815T001-r6-negative.Uovj8B/`。三类副本均同步刷新 artifact
  records/core hash，source/archived verifier 共 `6/6` 次 `rc=2`：
  - 不存在的 `outcome_shadow/accepted_stage1` 根目录：
    `accepted Stage 1 resolved root drift`；
  - 内容与正式 Stage 1 完全相同的 clone root：
    `accepted Stage 1 resolved root drift`；
  - 控制账本路径加入 `consumption_ledgers/../`：
    `accepted Stage 1 canonical path drift`。
- 正式包 source CLI、archived CLI `--verify-only` 均通过，报告：
  - `artifact_count=15`；
  - `candidate_membership_rows=463612`；
  - `confirmed_rows=267554`；
  - `sensitivity_membership_rows=463612`；
  - `session_count=3`；
  - core SHA256
    `cc6a10cc022d97a4fda01a9988a3dfd7a8c713a243d6d8eaacb6a21d309a095c`。
- 正式包、Build A
  `/tmp/0815T001-r6-density-a.VfHvoG/package` 和 Build B
  `/tmp/0815T001-r6-density-b.Gc72PR/package` 均为：
  - `16` files；
  - `19,834,631` bytes；
  - full-directory inventory SHA256
    `636d5a9f4a9940ac6a01c9b1db4eec3d0224a0d0574d4195486e1960ba47e2af`；
  - core package SHA256
    `cc6a10cc022d97a4fda01a9988a3dfd7a8c713a243d6d8eaacb6a21d309a095c`。
- Formal / Build A / Build B 的全部 relative paths、bytes 和逐文件
  SHA256 完全一致；source CLI 对 A/B 的 `--compare-to` 均
  `identical=true`。
- Archived verify-only 前后正式包 inventory 均为上述值；无
  `__pycache__`、`.pyc` 或新增路径。
- Source inventory before/after 继续为 `123` files、
  `169,677,903` bytes、SHA256
  `94bad85bfd4981b351f84c53628099468ec27f13d308402ac7125a9d582a6644`。
- Accepted Stage 1 actual/binding before/after 继续为 `14` files、
  `1,401,387` bytes、SHA256
  `c540cc056313716b3bdd2b9c0fe076cda15a7f152b399ae6a1283b3aa8aa6590`；
  core 保持
  `9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96`。
- 相对 Round 6 QA 对象只变化：
  `density_manifest.json`、`frozen_density_contract.json`、
  `runtime_source/cross_exchange_trigger_density_admission.py`。
- 正式 manifest boundary 七项继续全部为 `false`。

unchanged research evidence：
- 三 session 计数仍为 Jul30 `268522 / 141768`、Aug03
  `127622 / 82533`、Aug04 `67468 / 43253`。
- Cluster/flow/Family A overlap-block 仍分别为：
  - Jul30：`39928 / 10536 / 9`；
  - Aug03：`47193 / 23397 / 232`；
  - Aug04：`23113 / 9451 / 6`。
- Bartlett candidate/confirmed ESS 仍分别为：
  - Jul30：`1096.308299 / 837.102917`；
  - Aug03：`2195.991765 / 2085.852134`；
  - Aug04：`275.582545 / 263.249320`。
- Trigger 仍是近连续过程；row count 不能解释为 IID support。

done：
- Round 6 P1 已在同一 `0815T001` 范围内 bounded repair。
- 修复版正式包已发布到
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage02_density/`。
- 当前任务重新进入 `待验收`。Round 7 独立 QA `已通过` 前，Stage 3
  detector parity 及所有后续阶段继续锁定。

blockers：
- 无

commit：
- 无

提交信息：
- 无
