# 0729T002 业务执行回报

执行线程：
- 业务执行线程

任务ID：
- 0729T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `latency-probe/ec2_spread_hunt.py`
- `latency-probe/test_ec2_hunt.py`
- `docs/evidence/ec2_latency_hunt_20260729/runtime_control_source.tar.gz`
- `docs/evidence/ec2_latency_hunt_20260729/execution_manifest.json`
- `docs/evidence/ec2_latency_hunt_20260729/README.md`
- `.workflow/tasks/0729T001.md`
- `.workflow/tasks/0729T002.md`
- `.workflow/reports/0729T001-business.md`
- `.workflow/reports/0729T002-business.md`
- controller workflow files

action：
- 在任何修复前封存 T001 runtime control source archive，archive SHA 为
  `b609a836...3480`；归档内 6 个文件逐一匹配 T001 runtime source
  hashes。
- `finalize_winner` 现在先写 winner tags、启用 termination protection
  并读回确认 `true`，随后才允许 terminate losers 和删除 bucket。
- protection 未生效时立即 fail closed，任何 loser terminate 不得发生。
- final receipt 同时保留 protection-before-cleanup 和最终 protection。
- 新增 success finalizer regression，断言 mutation 顺序、六台 loser
  集合、winner 排除、`awsserver1` 排除、唯一 active winner 和 retained
  placement group。
- 新增 protection-failure regression，断言失败时没有 loser mutation。
- rack-level 测试名已修正；两处 EOF 空行已清理。
- execution manifest 升级为 v2，明确区分 T001 runtime source hashes 与
  T002 post-QA repair source hashes。
- 未执行任何 AWS mutation，未重跑 hunting，未修改当前两台服务器。

verify：
- Python focused：`24 passed`。
- Rust focused：`15 passed`；clippy 通过。
- Python compile、shell syntax、Rust fmt：通过。
- runtime source archive 内 6 个 hashes 与 T001 manifest 一致。
- T002 repaired source hashes 与当前文件一致。
- worktree diff check：通过。
- staged diff check 在重新 stage T002 文件后执行并要求通过。

done：
- T001 P1 finalizer 顺序缺陷已修复。
- T001 P2 success test coverage 与 staged diff hygiene 缺陷已修复。
- T001 P3 host/rack 测试命名缺陷已修复。
- T001 runtime provenance 未被 T002 离线代码冒充。

blockers：
- 无任务内阻塞。
- T002 不改变 T001 历史实际 mutation 顺序，只保证未来 controller 正确。
- CloudTrail `LookupEvents` 权限缺失仍为独立审计限制。

commit：
- 无

提交信息：
- 无
