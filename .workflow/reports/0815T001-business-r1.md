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
- 无

files：
- `examples/hyperliquid/cross_exchange_trigger_density_admission.py`
- `examples/hyperliquid/test_cross_exchange_trigger_density_admission.py`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage02_density/`
- `.workflow/tasks/0815T001.md`
- `.workflow/reports/0815T001-business-r1.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 修复首轮 QA P1：verify-only 现在把 observed membership session set、
  Candidate/confirmed counts 和 segment set 精确对比
  `FROZEN_SESSION_COUNTS` 与 `FROZEN_SEGMENT_EVIDENCE_LABELS`。不能再从
  membership、small CSV 和 manifest 同步删除一个完整 frozen session。
- 修复首轮 QA P1：冻结 source inputs 和 accepted Stage 1 的完整
  `file_count/total_bytes/inventory_sha256`。verify-only 在 before/after
  自洽检查之外，还必须精确匹配外部冻结 inventory contract；删除任意
  path/role 记录并同步重哈希会 fail closed。
- 修复首轮 QA P1：冻结每 session 的 cluster/flow/overlap、boundary、
  merge/non-merge 和 recovery/decision 分类精确结果。verify-only 要求
  JSON key/value 类型、总数、merge mapping 和 frozen result 全部一致，
  不再把 published JSON 自己当 expected value。
- 修复首轮 QA P2：evidence labels 收敛到主方案冻结枚举。
  Jul30 session aggregate 使用 `historical_discovery`，caveat 明确其
  segments 0001-0004 为 discovery、0005-0008 为
  `historical_internal_validation`；Aug03 使用
  `historical_transfer` 且继续 `formal_eligible=false`；Aug04 使用
  `historical_consumed_validation`。
- 把 exact segment evidence mapping、inventory contract、merging result
  和 session evidence 写入 canonical frozen contract；同 schema 下的
  missing/extra/value drift 均 fail closed。
- 新增首轮 QA 同构的 coherent-rehash 永久回归：
  complete-session deletion、source/Stage 1 binding omission、
  forged recovery/decision categories，以及 frozen taxonomy 检查。
- 从只读输入重新原子发布正式包，并重新完成两次隔离全量构建。
- 未修改 density、inter-trigger、candidate/sensitivity membership、
  merging summary、ESS 或报告数值；未读取 Aug07 event rows，未运行
  detector parity、Episode v3、outcome/model/PnL/actionability。

verify：
- 完整聚焦回归通过：
  `80 passed in 7.22s`。
- admission coherent-rehash/taxonomy targeted 回归通过：
  `4 passed`。
- Ruff、compileall、CLI help 和 `git diff --check` 全部通过。
- 当前源码入口正式包 `--verify-only` 通过。
- 正式包内 archived CLI `--verify-only` 通过；运行前后没有
  `__pycache__`、`.pyc` 或额外路径。
- 正式包、`/tmp/0815T001-r1-density-a.jFmowo/package` 和
  `/tmp/0815T001-r1-density-b.ogtFnV/package` 均为 `16` files、
  `19,814,426` bytes，逐 path/bytes/SHA256 完全一致。
- 三次构建 core package SHA256 均为
  `532448d498c8206cd5669d26f1c459b46c3c0505690b15957925182b3f98ee03`。
- 三次 full-directory inventory SHA256 均为
  `1cd78478575cff2eb4fe8bf167f570a1c8412126ab4219cfd05f76b152d3f977`。
- 三次 source inventory before/after 均为 `123` files、
  `169,677,903` bytes、SHA256
  `94bad85bfd4981b351f84c53628099468ec27f13d308402ac7125a9d582a6644`。
- 三次 accepted Stage 1 inventory before/after 均为 `14` files、
  `1,401,387` bytes、SHA256
  `c540cc056313716b3bdd2b9c0fe076cda15a7f152b399ae6a1283b3aa8aa6590`。
- 与首轮 formal package 相比，只有
  `density_manifest.json`、`frozen_density_contract.json`、
  `runtime_source/cross_exchange_trigger_density_admission.py` 和
  `trigger_density_by_session.csv` 发生变化；其余 `12` 个文件逐
  bytes/SHA256 不变。
- 首轮独立 QA 已确认的 counts/rates/quantiles/window coverage/
  cluster-flow-overlap/60s catalog/Bartlett ESS/sensitivity selected counts
  因此全部保持不变。

done：
- 首轮 QA P1 x3 和 P2 x1 已在同一 `0815T001` 范围内 bounded repair。
- 修复版正式包已发布到
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage02_density/`。
- 当前任务重新进入 `待验收`。独立 re-QA `已通过` 前，Stage 3
  detector parity 及所有后续阶段继续锁定。

blockers：
- 无

commit：
- 无

提交信息：
- 无
