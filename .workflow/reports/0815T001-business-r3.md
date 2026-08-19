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
- 请由新的独立 QA 线程执行第四轮验收，不沿用业务或 Round 3 QA 结论。

files：
- `examples/hyperliquid/cross_exchange_trigger_density_admission.py`
- `examples/hyperliquid/test_cross_exchange_trigger_density_admission.py`
- `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage02_density/`
- `.workflow/tasks/0815T001.md`
- `.workflow/reports/0815T001-business-r3.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 修复 Round 3 QA P1：新增
  `candidate_trigger_projection_v1`，按 session 冻结完整 Candidate
  authoritative/structural projection 的 exact row count 与 canonical
  SHA256。
- Projection fields 精确覆盖：
  `session_id`、`candidate_id`、`primary_episode`、`rejection_reason`、
  `aggressor_side`、`direction_sign`、`shock_ts_ns`、
  `decision_ts_ns`、`impact_ratio`、`pre_state_ts_ns`、
  `pre_state_age_ms`、`segment_id`、`connection_epoch_id`、
  `segment_start_ts_ns`、`segment_end_ts_ns`。
- verify-only 从 `candidate_episode_membership.csv.gz` 逐行重算每 session
  projection；完成既有逐行语义、count、segment 和 aggregate 验证后，
  再精确比较外部 runtime-source anchor：
  - Jul30：`268522` rows，
    `155d3b74898e10d56344f07808da4cea0abcec2696bd9bb6174e32c1bb8e2416`；
  - Aug03：`127622` rows，
    `20028e23d1e4393a819c1d2882ce212a5cba96d8b57d4e7b55bd51d9a3d313d6`；
  - Aug04：`67468` rows，
    `7266e1ac9f3654b74b4802eca01b31e1332c5499cd2b91c24cfc4f3938796bcb`。
- Projection version、字段顺序和三个 session anchors 同时写入 canonical
  frozen contract；missing/extra session、row count 或 SHA drift 均 fail
  closed。
- 修复 Round 3 QA P2：新增统一 per-row CSV cell closure。所有 plain/gzip
  reader 在读取字段前要求：
  - row key set 精确等于 header；
  - 不存在 `None` unnamed extra-cell key；
  - 每个 expected field 的 value 都不为 `None`。
- strict closure 覆盖 `input_bindings.csv`、五个 small CSV、
  `candidate_episode_membership.csv.gz` 和
  `trigger_density_sensitivity_membership.csv.gz`。
- 新增 permanent hostile regressions：
  - exact rejected `rejection_reason` 替换；
  - sensitivity threshold bucket 不变的 `impact_ratio` 漂移；
  - `pre_state_ts_ns` 与 `pre_state_age_ms` 成对自洽漂移；
  - 八个 CSV 入口分别执行 leading/middle/trailing extra cell 和 missing
    cell，共 `32` 组 row-width 攻击。
- 从冻结只读输入重新原子发布正式包，并完成两个不同 `/tmp` 目录的
  独立完整构建。
- 未改变 detector、merging、sensitivity、ESS 或任何研究结果。与 Round 3
  QA 所验收的 R2 Build A 相比，仅 `density_manifest.json`、
  `frozen_density_contract.json` 和 archived admission verifier 变化；
  其余 `13` 个文件逐字节不变。

verify：
- Admission suite：`79 passed in 1.79s`。
- Stage 2 四文件完整 focused regression：`130 passed in 7.94s`。
- Reused hierarchy recovery regressions：
  `3 passed, 19 deselected in 0.03s`。
- Ruff、compileall、CLI help 和 `git diff --check` 全部通过。
- 现有正式 membership 独立重算的三个 projection row count/SHA 与冻结
  anchors 精确相等。
- Round 3 production-sized 原始攻击在
  `/tmp/0815T001-r3-hostile.YwLmEZ/` 复现：
  - `forged-rejection-reason/package`：把
    `jul30:segment_0001:6` 改为 `forged_rejection_reason` 并同步重哈希；
    source/archived CLI 均 `rc=2`，命中 Jul30 projection SHA drift；
  - `impact-ratio-drift/package`：把 `jul30:segment_0001:1` 从
    `7.516129032258064` 改为 `7.517129032258064` 并同步重哈希；
    两个 CLI 均 `rc=2`，命中 Jul30 projection SHA drift；
  - `density-extra-cell/package`：在首行附加未命名
    `future_outcome=forged` cell 并同步重哈希；两个 CLI 均 `rc=2`，
    命中 `row cell/schema drift at row 2`。
- 正式包 source CLI 与 archived CLI `--verify-only` 均通过，报告：
  - `artifact_count=15`；
  - `candidate_membership_rows=463612`；
  - `confirmed_rows=267554`；
  - `sensitivity_membership_rows=463612`；
  - `session_count=3`；
  - core SHA256
    `982011eae0851112747ee8b98a9308edf3843c2ac7505105ee2c750951dbda02`。
- Archived CLI verify-only 前后正式包完整 inventory 一致；无
  `__pycache__`、`.pyc` 或新增路径。
- 正式包、Build A
  `/tmp/0815T001-r3-density-a.kKplZL/package` 和 Build B
  `/tmp/0815T001-r3-density-b.46adRv/package` 均为：
  - `16` files；
  - `19,826,238` bytes；
  - full-directory inventory SHA256
    `5bc042e695cf61938dd1687124a4140c6e99a1be287279b95a53b1df1d6607dc`；
  - core package SHA256
    `982011eae0851112747ee8b98a9308edf3843c2ac7505105ee2c750951dbda02`。
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
- Round 3 QA P1/P2 已在同一 `0815T001` 范围内 bounded repair。
- 修复版正式包已发布到
  `local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage02_density/`。
- 当前任务重新进入 `待验收`。Round 4 独立 QA `已通过` 前，Stage 3
  detector parity 及所有后续阶段继续锁定。

blockers：
- 无

commit：
- 无

提交信息：
- 无
