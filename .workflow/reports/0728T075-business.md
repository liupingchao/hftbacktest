# 0728T075 业务执行回报

执行线程：
- 业务执行线程

任务ID：
- 0728T075

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `latency-probe/ec2_hunt_orchestrator.py`
- `latency-probe/test_ec2_hunt.py`
- `docs/evidence/ec2_latency_hunt_20260728/README.md`
- `docs/evidence/ec2_latency_hunt_20260728/control_canary.json`
- `docs/evidence/ec2_latency_hunt_20260728/execution_manifest.json`
- workflow controller files

action：
- `execute()` 捕获任意 KeyboardInterrupt 时显式设置
  `interrupted=true`，再写 failure journal 并执行 finally cleanup。
- CLI `--task` 改为 required，禁止默认身份随任务轮换后过期。
- 新增直接 KeyboardInterrupt 和缺少 `--task` 的回归测试。
- README final Canary identity 修正为 T074。
- execution manifest 区分 T074 Canary runtime source 与最终 T075
  offline-repaired source。

verify：
- Python：`18 passed`。
- Rust：`15 passed`；fmt/clippy 通过。
- 直接 KeyboardInterrupt：journal `interrupted=true`，cleanup 恰好一次。
- SIGTERM handler：journal `interrupted=true`，cleanup 恰好一次。
- parse_args 缺少 `--task`：SystemExit，stderr 包含 `--task`。
- `git diff --check`、`git diff --cached --check`：通过。
- 未执行 AWS mutation；复用 QA 已接受的 T074 final Canary 和零遗留。

done：
- T074 QA 的两个 P2 和一个 P3 均已修复。
- Goal 2 的 10-host 数据、build provenance、fail-closed control path、
  interruption audit 和 task identity 已具备最终 QA 条件。

blockers：
- 无。

commit：
- 无

提交信息：
- 无
