执行线程：
- 业务线程-repository-alignment

任务ID：
- 0731T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- 当前仓库 Git 工作树
- `macmini-tunnel:/Users/liu/Documents/hftbacktest`
- `/tmp/hftbacktest-sync-20260731`
- `pre-macmini-sync-20260731`

action：
- 确认本地原始 `HEAD=e11f0437` 且工作树干净，Mac 为
  `HEAD=aee70284` 且包含 staged、unstaged 和 untracked 改动。
- 创建本地恢复分支 `pre-macmini-sync-20260731`。
- 在 Mac `/tmp` 导出并校验 commit bundle、staged binary patch、
  unstaged binary patch、60 个 Git-visible untracked 文件归档和状态清单。
- 本地快进到 `aee70284`，按 staged、unstaged、untracked 三层恢复 Mac
  工作树，保留原暂存边界。
- 用户收缩范围后停止 ignored evidence 计划；带 `-n` 的 rsync dry-run
  未写入任何数据，约 4 GB `local_live_analysis` 证据未传输。

verify：
- 本地与 Mac `HEAD` 均为
  `aee70284ad67a5aa5e80bc5c7c0c1e6347de91ce`。
- 排除本轮记录前，两边 `git status --porcelain=v1` 完全一致。
- staged patch SHA-256：
  `dcf980040ae3b41dfac780c784728e97050971415292e0673b659590b839c0b5`。
- unstaged patch SHA-256：
  `a0aa104764b17748387a3835f06fcb236ef9b4448ba384d51a4ef1b762367bfb`。
- Mac 的 60 个 Git-visible untracked 文件逐文件 SHA-256 全部通过。
- Cross-exchange Python focused：`103 passed`。
- Latency-probe Python：`24 passed`。
- `cargo test -p latency-probe`：`15 passed`。

done：
- 当前仓库已具备 Mac 上的提交、代码、关键文档和完整 Git 工作树状态。
- ignored `local_live_analysis`、构建产物、缓存和秘密文件未同步。
- 双端恢复快照保留在 `/tmp/hftbacktest-sync-20260731`。

blockers：
- 无

commit：
- 无

提交信息：
- 无
