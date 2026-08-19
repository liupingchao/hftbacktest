# 业务线程回报

执行线程：
- 业务线程-python/research

任务ID：
- 0815T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 请由新的独立 QA 线程执行第三轮验收，不沿用业务结论。

files：
- `examples/hyperliquid/cross_exchange_trigger_density_admission.py`
- `examples/hyperliquid/test_cross_exchange_trigger_density_admission.py`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage02_density/`
- `.workflow/tasks/0815T001.md`
- `.workflow/reports/0815T001-business-r2.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 修复第二轮 QA P1：为五个 small CSV artifact 构造外部冻结的完整
  expected primary-key set：
  - density：`session_id x Family A/B`；
  - inter-trigger：`session_id x Family A/B x
    all/same_side/opposite_side`；
  - episode summary：每个 frozen session 恰好一行；
  - sensitivity summary：`session_id x Family A/B x 六个 named
    sensitivities`；
  - ESS：每 session 的结构 support keys，加每 population 的
    session-total 与逐 segment Bartlett keys。
- Production verify-only 对五个 artifact 全部要求
  `len(rows) == len(lookup) == len(expected_keys)` 且 actual key set 与
  expected key set 精确相等。duplicate shadow row、缺失 key 和额外 unique
  key 均 fail closed。
- 新增 `15` 组 coherent-rehash 永久回归：
  - 五个 artifact 的 duplicate forged row 分别位于合法行之前和之后；
  - 五个 artifact 分别追加一个额外 unique session/population/side/
    sensitivity/metric key。
- 旧 complete-session deletion 回归现在由更早、更强的 density
  primary-key closure invariant 截获；测试明确绑定新的失败原因。
- 从冻结只读输入重新原子发布正式包，并完成两个不同 `/tmp` 目录的
  独立完整构建。
- 未改变任何研究数值或 membership。与第二轮 QA 所验收的 R1 Build A
  比较，只有 `density_manifest.json` 和
  `runtime_source/cross_exchange_trigger_density_admission.py` 发生变化，
  其余 `14` 个文件逐字节不变。

verify：
- Admission 专项：
  `44 passed in 0.81s`。
- Stage 2 四文件完整 focused regression：
  `95 passed in 6.86s`。
- Reused hierarchy recovery helper 原始 regression：
  `3 passed, 19 deselected in 0.03s`。
- Ruff、compileall、CLI help 和 `git diff --check` 全部通过。
- Source CLI 与正式包 archived CLI `--verify-only` 均通过，报告：
  - `artifact_count=15`；
  - `candidate_membership_rows=463612`；
  - `confirmed_rows=267554`；
  - `sensitivity_membership_rows=463612`；
  - `session_count=3`；
  - `core_package_sha256=6ee9bc7f78d484b42243f74d9973ecded8062136a9b5ab741fed66c42735353a`。
- Archived CLI verify-only 前后，正式包完整文件 inventory 一致；没有
  `__pycache__`、`.pyc` 或新增路径。
- 正式包、Build A
  `/tmp/0815T001-r2-density-a.5tyH0q/package` 和 Build B
  `/tmp/0815T001-r2-density-b.wWx3mh/package` 均为：
  - `16` files；
  - `19,819,261` bytes；
  - full-directory inventory SHA256
    `2304dfc78265ee32bcf659f7af3fa876219beb37e15c424f4be790c5353c353a`；
  - core package SHA256
    `6ee9bc7f78d484b42243f74d9973ecded8062136a9b5ab741fed66c42735353a`。
- Formal / Build A / Build B 的全部 relative paths、bytes 和逐文件
  SHA256 完全一致；source CLI `--compare-to` 也报告 `identical=true`。
- 三次构建的 source inventory before/after 均为 `123` files、
  `169,677,903` bytes、SHA256
  `94bad85bfd4981b351f84c53628099468ec27f13d308402ac7125a9d582a6644`。
- 三次构建的 accepted Stage 1 inventory before/after 均为 `14` files、
  `1,401,387` bytes、SHA256
  `c540cc056313716b3bdd2b9c0fe076cda15a7f152b399ae6a1283b3aa8aa6590`。
- Stage 1 core SHA256 仍为
  `9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96`。
- 正式 manifest 继续声明并验证：
  `aug07_event_rows_read=false`、`response_fields_read=false`、
  `outcome_fields_read=false`、`model_or_score_run=false`、
  `pnl_or_actionability_run=false`、`new_collection=false`、
  `private_order_cancel_access=false`。

done：
- 第二轮 QA 新增 P1 已在同一 `0815T001` 范围内 bounded repair。
- 修复版正式包已发布到
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage02_density/`。
- 当前任务重新进入 `待验收`。第三轮独立 QA `已通过` 前，Stage 3
  detector parity 及所有后续阶段继续锁定。

blockers：
- 无

commit：
- 无

提交信息：
- 无
