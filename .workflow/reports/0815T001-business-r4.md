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
- 请由新的独立 QA 线程执行第五轮验收，不沿用业务或前四轮 QA 结论。

files：
- `examples/hyperliquid/cross_exchange_trigger_density_admission.py`
- `examples/hyperliquid/test_cross_exchange_trigger_density_admission.py`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage02_density/`
- `.workflow/tasks/0815T001.md`
- `.workflow/reports/0815T001-business-r4.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 修复 Round 4 QA P1：把 projection 升级为
  `candidate_membership_projection_v2`，字段序列精确等于完整
  `CANDIDATE_MEMBERSHIP_FIELDS`。
- v2 在 Round 3 已冻结的 authoritative/structural fields 后继续纳入：
  `cluster_id`、`continuous_flow_episode_id`、`overlap_block_id` 和
  `window_end_ts_ns`。verify-only 因而精确锚定每个 candidate 的完整
  Stage 2 structural membership，而不再只验证 aggregate group counts。
- v2 每 session exact anchors：
  - Jul30：`268522` rows，
    `cd731b5b011183366fd3cb04e0809d082a26bde85d7f1446ff4754016c3551f9`；
  - Aug03：`127622` rows，
    `04cc9a13c85e530291bcf3c584e78ea029ed67fe1653bcdd7efa13e14294fda0`；
  - Aug04：`67468` rows，
    `bf5911989a07f9b1242b9eef12e72a40093549f98c516c846df6caae353db820`。
- 修复 Round 4 QA P2：新增
  `FROZEN_MEMBERSHIP_SESSION_ORDER =
  ("jul30", "aug03", "aug04")`。verify-only 在流式消费两个 membership
  gzip 时记录完整 session block transition，并要求 observed order 与
  frozen order 精确相等。
- Projection version、完整 19 字段、三个 session anchors 和 session order
  全部进入 canonical frozen contract；source/archived verifier 消费同一
  runtime constants。
- 新增永久回归：
  - 对合法格式的 `cluster_id`、`continuous_flow_episode_id`、
    `overlap_block_id` 分别做单行 coherent-rehash relabel，均须命中 v2
    projection SHA；
  - 同步交换两个 membership gzip 的完整 fixture session blocks，须命中
    frozen session-order invariant。
- 从冻结只读输入重新原子发布正式包，并完成两个不同 `/tmp` 目录的
  独立完整构建。
- 未更改 producer merging mapping、detector、sensitivity、ESS 或任何
  研究结果。相对 Round 4 QA 对象仍仅有 manifest、frozen contract 和
  archived verifier 三个 admission 文件变化；其余 `13` 个文件逐字节
  不变。

verify：
- Admission suite：`83 passed in 1.86s`。
- Stage 2 四文件完整 focused regression：`134 passed in 8.20s`。
- Reused hierarchy recovery regressions：
  `3 passed, 19 deselected in 0.03s`。
- Ruff、compileall、CLI help 和 `git diff --check` 全部通过。
- v2 projection 现有正式 membership 重算结果与三个 frozen anchors
  精确相等，projection fields 与 `CANDIDATE_MEMBERSHIP_FIELDS` 精确相等。
- Round 4 production-sized 原始缺陷在
  `/tmp/0815T001-r4-hostile.Vzp9Au/` 复现：
  - `singleton-cluster-relabel/package`：把 Jul30 singleton cluster
    `segment_0006-0-C004886` 改为合法未使用 ID
    `segment_0006-0-C999999`，group count/quantile 不变并同步重哈希；
    source/archived CLI 均 `rc=2`，命中 Jul30 v2 projection SHA drift；
  - `swap-session-blocks/package`：同步把两个 membership gzip 改为
    `aug03 -> jul30 -> aug04` 并刷新全部 hashes；两个 CLI 均 `rc=2`，
    精确报告 expected `jul30,aug03,aug04` 与 observed
    `aug03,jul30,aug04` 的 session-order drift。
- 正式包 source CLI 与 archived CLI `--verify-only` 均通过，报告：
  - `artifact_count=15`；
  - `candidate_membership_rows=463612`；
  - `confirmed_rows=267554`；
  - `sensitivity_membership_rows=463612`；
  - `session_count=3`；
  - core SHA256
    `536229d7a08db6048887cac29093ff137856424655563b578b361f6b37ac1a6f`。
- Archived CLI verify-only 前后正式包完整 inventory 一致；无
  `__pycache__`、`.pyc` 或新增路径。
- 正式包、Build A
  `/tmp/0815T001-r4-density-a.Ok1hG1/package` 和 Build B
  `/tmp/0815T001-r4-density-b.5XtMXE/package` 均为：
  - `16` files；
  - `19,827,566` bytes；
  - full-directory inventory SHA256
    `bbea54622aebb4ef06f1f29a5ca4308a1a516fcaee61a2488f17a14c49276ac1`；
  - core package SHA256
    `536229d7a08db6048887cac29093ff137856424655563b578b361f6b37ac1a6f`。
- Formal / Build A / Build B 的全部 relative paths、bytes 和逐文件
  SHA256 完全一致；source CLI `--compare-to` 报告 `identical=true`。
- 三次构建 source inventory before/after 均为 `123` files、
  `169,677,903` bytes、SHA256
  `94bad85bfd4981b351f84c53628099468ec27f13d308402ac7125a9d582a6644`。
- 三次构建 accepted Stage 1 inventory before/after 均为 `14` files、
  `1,401,387` bytes、SHA256
  `c540cc056313716b3bdd2b9c0fe076cda15a7f152b399ae6a1283b3aa8aa6590`。
- Stage 1 core SHA256 保持
  `9c2756f966a1a05816f9e6a91d7361d76cfea10fc3c587944ca2f0c1d2eb7c96`。
- 正式 manifest boundary 七项继续全部为 `false`。

done：
- Round 4 QA P1/P2 已在同一 `0815T001` 范围内 bounded repair。
- 修复版正式包已发布到
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage02_density/`。
- 当前任务重新进入 `待验收`。Round 5 独立 QA `已通过` 前，Stage 3
  detector parity 及所有后续阶段继续锁定。

blockers：
- 无

commit：
- 无

提交信息：
- 无
