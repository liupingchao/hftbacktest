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
- 请由新的独立 QA 线程执行第八轮验收，不沿用业务或前七轮 QA 结论。

files：
- `examples/hyperliquid/cross_exchange_trigger_density_admission.py`
- `examples/hyperliquid/test_cross_exchange_trigger_density_admission.py`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage02_density/`
- `.workflow/tasks/0815T001.md`
- `.workflow/reports/0815T001-business-r7.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 修复 Round 7 QA P2：Stage 1 scope 的 canonical-path 检查保留原始
  CSV 字符串，并依次要求：
  - path 为绝对路径；
  - raw text 精确等于 `str(Path(raw_text))`；
  - raw text 精确等于 `str(Path(raw_text).resolve())`；
  - path 位于 frozen Stage 1 root；
  - path 与 frozen root/relative-path join 精确一致。
- 该规则使 `/./`、重复 `/`、尾随 `/`、`..`、symlink 解析差异和其他
  等价冗余文本只能有一个规范表示。
- 新增永久回归覆盖 raw `/./` 和 duplicate separator；既有 `..`、
  root substitution、builder root 和 actual Stage 1 bytes 回归继续通过。
- 从相同冻结只读输入重新原子发布正式包，并完成两个不同 `/tmp`
  目录的独立完整构建。
- Canonical frozen contract 已在 Round 6 明确声明
  `canonical_resolved_root_and_absolute_paths`，本轮使 runtime 行为兑现
  该既有合同，因此 contract bytes 未变化。
- 未更改 detector、candidate population、merging mapping、sensitivity、
  ESS 或任何研究数值。

verify：
- Admission suite：`96 passed in 2.77s`。
- Stage 2 四文件完整 focused regression：`147 passed in 8.81s`。
- Reused hierarchy recovery regressions：
  `3 passed, 19 deselected in 0.09s`。
- Ruff、外置 pycache compileall、CLI help 和 scoped
  `git diff --check` 全部通过。
- Source/archived `--help` 均为 `26` 行且逐字节一致。
- Production-sized Round 7 修复证据位于
  `/tmp/0815T001-r7-negative.CFG0Hn/`。三类副本均同步刷新 artifact
  records/core hash，source/archived verifier 共 `6/6` 次 `rc=2`：
  - root 后加入 `/./`；
  - root 后加入重复 `/`；
  - 文件路径加入尾随 `/`；
  三类均命中 `accepted Stage 1 canonical path drift`。
- 正式包 source CLI、archived CLI `--verify-only` 均通过，报告：
  - `artifact_count=15`；
  - `candidate_membership_rows=463612`；
  - `confirmed_rows=267554`；
  - `sensitivity_membership_rows=463612`；
  - `session_count=3`；
  - core SHA256
    `7b3d06c3225f77c9929cdd3fa40d69c0866d83dbb75dd584dcb7db6db22ad3f8`。
- 正式包、Build A
  `/tmp/0815T001-r7-density-a.hr0lg4/package` 和 Build B
  `/tmp/0815T001-r7-density-b.RTLgQF/package` 均为：
  - `16` files；
  - `19,834,728` bytes；
  - full-directory inventory SHA256
    `bdade16c53aed7bba54fdb9408ba3a8a03036d183720f589bda2a762448a3833`；
  - core package SHA256
    `7b3d06c3225f77c9929cdd3fa40d69c0866d83dbb75dd584dcb7db6db22ad3f8`。
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
- 相对 Round 7 QA 对象只变化：
  `density_manifest.json` 和
  `runtime_source/cross_exchange_trigger_density_admission.py`。
  `frozen_density_contract.json` 与所有研究 artifacts 逐字节不变。
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
- Round 7 P2 已在同一 `0815T001` 范围内 bounded repair。
- 修复版正式包已发布到
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage02_density/`。
- 当前任务重新进入 `待验收`。Round 8 独立 QA `已通过` 前，Stage 3
  detector parity 及所有后续阶段继续锁定。

blockers：
- 无

commit：
- 无

提交信息：
- 无
