# 0831T001 Plan Amendment Round 18

执行线程：
- SKHYNIX Trade-Led Depth-Follower Q0 总控

任务ID：
- 0831T001

状态：
- 待验收

是否进行QA验收：
- 否

目的：
- 关闭 implementation readiness round 6 的三个 P1，使 PRE_BLOCKER
  ordering、recovery-start immutable identity 和 control publication path
  kind 具有可执行且非自证的唯一语义。

amendment：
1. PRE_BLOCKER 必须先从 canonical validated receipt union、tracked-copy
   equality 和本地 Git 状态确定 action phase，再按 G01-G07 评价；无效
   receipt 不得推进 proof-stage，A01-A11 只能在 G evaluation 后运行。
2. 所有 durable control final target 与 sibling temporary 均使用 no-follow
   `lstat`；final target 必须是 regular non-symlink，调用者禁止在 publication
   前解析 final path component。
3. `recovery_start.json` 增加 macOS sealed-authority 语义：
   - exact-byte hard-link 后先 `chmod 0444`；
   - 设置 `UF_IMMUTABLE`；
   - fsync parent 并重新 no-follow 验证；
   - seal 完成前禁止任何 recovery state mutation。
4. 如果在 seal 前崩溃，当前 snapshot 必须仍是原始状态，restart 只能独立
   重建相同字节并完成 seal。已 seal 文件缺 mode/flag、是 symlink 或 bytes
   漂移均 fail closed，禁止修复或重算 self-hash。

platform evidence：
- exact formal runtime 是当前 macOS 工作区。
- 临时探针确认 `os.chflags`、`stat.UF_IMMUTABLE`、`st_flags` 可用。
- `0444 + UF_IMMUTABLE` 后普通写入返回 `operation not permitted`。
- 探针已解除 flag 并删除，不属于任务产物。

new authority：
- execution plan SHA256 / blob：
  `cd958a033ae4b5b4bfb862b6b10ce679c913d958043cd892230330920627c920`
  / `92bd9dc9cc5de16155dd05cf199611c73c9a8531`
- task SHA256 / blob：
  `3205ec736acc02e21f98d9ec3f866612fbeb243a437107c7da7f6e3f5455013a`
  / `e05583e43247c4ce3b3771f047d0b5069e9415f3`
- surface SHA256 / blob：
  `6ed73252048a4e501580dac1a3b810f2a4f649d02f06dc361a6b71262ccb8e2a`
  / `a2a83d89727346ecff184be261f64312c19aadcd`
- fixture truth：保持不变。

boundary：
- current date：2026-08-31。
- historical cache / future outcome：未访问。
- implementation tag、claim、controller、formal root、receipt 和 task tag：
  均未创建。
- implementation readiness round 6 已独立记录为 FAIL。

结论：
- 请求独立 plan review。
- review 通过前 authority-dependent implementation 和 formal 保持锁定。

提交信息：
- `plan: amend 0831T001 sealed recovery authority`
