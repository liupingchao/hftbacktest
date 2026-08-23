# 0822T002 amdserver QA Sync Report

执行线程：
- 总控

任务ID：
- 0822T002

状态：
- 待验收

同步时间：
- `2026-08-23T10:43:47Z`（星期日）

是否进行QA验收：
- 是

QA说明：
- 业务 candidate 已同步到独立 amdserver detached worktree，等待用户在
  amdserver 执行独立 QA。
- QA 不得调用 private、order 或 cancel endpoint；只验证已提交 source、
  sealed evidence、formal package 和 durable archive。

files：
- source worktree：
  `/home/molly/project/qa_worktrees/0822T002-candidate`
- formal package：
  `/home/molly/project/qa_worktrees/0822T002-candidate/`
  `local_live_analysis/skhynix_c6in_hyperliquid_execution_latency_0822T002`
- durable archive：
  `/home/molly/project/durable_archives/skhynix_c6in_latency/`
  `7d851ab161ec02c621dffff63ef2f3e962a2aa0c7ed8b4382b285df9534bb0df`
- task-scoped Ruff tools：
  `/home/molly/project/qa_envs/0822T002-tools`

action：
- 保留 amdserver 现有
  `/home/molly/project/hftbacktest` `cross-exchange` checkout 不变。
- 从其 Git object store 创建独立 detached worktree，并固定到 business
  candidate commit
  `6859d3fae7f6020daaa4e2067ff352324a13fbd3`。
- 从 worktree-external durable archive 复制 exact formal package 到
  checkout 的 Git-ignored `local_live_analysis/...` 路径。
- 在 checkout 外安装 task-scoped `ruff==0.12.0`；pytest 使用 amdserver
  Anaconda Python 自带的 `pytest 8.3.4`。

verify：
- detached checkout HEAD=
  `6859d3fae7f6020daaa4e2067ff352324a13fbd3`，ordinary Git status rows=`0`。
- Gate 0 通过：
  `15 surfaces / 15 negative mutations / 7 exit criteria`；
  Surface Matrix SHA256=
  `8c1cad9654868888c9baebaeb4c714d2f4ddec4458ee0a5304527f2a8a9b30a8`。
- Focused/inherited suite：
  `224 passed in 4.84s`。
- Hostile current/frozen：
  `35 cases / 70 executions / fail-open 0`。
- Ruff `0.12.0`：
  `All checks passed`。
- Fresh sealed L1 A/B rebuild 均与 formal package 的四个结果文件
  byte-identical。
- Formal package admission：
  `verified=true`、`zero_write=true`、`28` files、`682135` bytes、
  sample/reliability gates 均通过。
- Package inventory SHA256=
  `1750474bdd04e1ff5b4beaddf1d93c3e79177060abd2cf7e3c6bacad0876af43`；
  measurement manifest SHA256=
  `8ac3b362e8d64cbd81232eaf7ed5856bada63ece20408e0d0b3fb5f84c562afd`。
- archive/package `rsync -ainc --delete` 输出为空，exact-tree parity 通过。
- formal package 明确冻结
  `kernel_package_admission_portable=true`、
  `full_source_semantic_replay_portable=false`。amdserver QA 不得把
  Linux package admission 表述为 portable full source-semantic replay。

done：
- 0822T002 的 committed source、sealed evidence、formal package、
  task-scoped QA tools 和 durable archive 已在 amdserver 就绪。
- QA 可直接从
  `/home/molly/project/qa_worktrees/0822T002-candidate`
  开始，不需要切换或清理现有主 checkout。
- H0-B 继续锁定。

blockers：
- 无同步 blocker。

commit：
- business candidate：
  `6859d3fae7f6020daaa4e2067ff352324a13fbd3`

提交信息：
- `research: complete c6in latency measurement`
