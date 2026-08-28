# QA 验收结果

执行线程：
- QA验收线程

任务ID：
- 0828T008

状态：
- 已通过

更新时间：
- 2026-08-28 13:24 CST

验收线程：
- QA验收线程

验收对象：
- SKHYNIX Liquidity Break Onset A0 Execution 业务线程
- implementation/evidence commit：`c30cad90`
- business report commit：`044e6a1e`
- causal-boundary test remediation commit：`cfd9ef54`
- remediated report commit：`6ef4c019`

验收范围：
- 验收 Revision 2 contract SHA、source closure 和 raw-message causal
  replay。
- 验收 normalization calibration boundary、exact event onset、active
  state machine、outcome-blind controls 和 follow-up geometry。
- 验收 A0-0 至 A0-6 gates、canonical failure classification 和 A1
  authorization boundary。
- 验收 compact artifact closure、double-build determinism、test coverage
  和 Git path scope。

验收步骤：
1. 核对 contract SHA256 为
   `b69146b92411a425e9d78d9deb88ac5347ccc0b2ad5b6f2ac32167b9eecb1141`。
2. 核对 source manifest：29 captures、9 dates、35.9172 hours、raw hashes
   现场复核。
3. 核对 detector 只消费当前及过去消息，未读取 future midpoint 或
   best-price target。
4. 核对 anchors、active intervals、controls、matching、component support
   和 follow-up geometry。
5. 核对 failure classification 与 frozen gates 的映射。
6. 对两个 output roots 的 28 个 compact artifacts 做 path/size/SHA
   比较。
7. 执行 15 个 focused synthetic tests、ruff、Python compile 和
   `git diff --check`。
8. 核对 commits 未纳入已有无关 staged 文档。

实际结果：
- Source closure、zero-outcome boundary、detection geometry 和 follow-up
  geometry 通过。
- 340,068 anchors 对应 `9468.1098/hour`，超过上限 `300/hour` 约
  31.6 倍。
- Active interval p50 `176.3133ms`；204,918 个 interval 由
  `opposite_onset` 结束。
- non-floor local scale 至少覆盖两个 component 的 share 为
  `0.64298`，低于 `0.95`。
- Overall control common support `0.73678`；minimum date `0.59027`，
  低于 frozen gates。
- Gates：A0-0 pass、A0-1 pass、A0-2 fail、A0-3 pass、A0-4 fail、
  A0-5 fail、A0-6 pass。
- Canonical classification 为 `A0_anchor_near_continuous`。
- `A1_authorized=false`。
- Outcome ledger：future fields `[]`、targets materialized `false`、
  H0/H1 fitted `false`。
- 两个 output roots 的 28 个 compact artifacts 零差异；
  `run_manifest.json` SHA256 均为
  `1f5f875d882e21aacd1be14793c3d3bd8e35f96ee8a0921290b3e6c8f774cdd0`。
- Focused tests：15 passed；ruff、compile、diff check 全部通过。

验收结论：
- 已通过
- 结论说明：
  - A0 implementation/execution 与证据链正确闭合，并正确否证
    `LIQUIDITY_BREAK_ONSET_V1` 当前版本。QA 通过不表示 hypothesis
    通过；研究结论仍是 `A0_anchor_near_continuous`，A1 不获授权。

通过项：
1. Contract/source/replay closure 完整。
2. Causal ordering、state machine、controls 和 no-outcome boundary
   有实现及 synthetic test 证据。
3. Failure gates 与 canonical classification 一致。
4. Deterministic compact artifacts 闭合。
5. Git commits 路径受控，未提交无关 staged/untracked files。

不通过项：
1. 无执行或证据验收缺陷。

缺陷清单：
1. 无未修复实现缺陷。

阻塞项：
- A1 target support audit 被 A0 passing gate 阻塞。
- 当前 hypothesis 不允许通过调高 threshold、改变 release 或查看
  outcomes 后 rescue；任何后续变体必须注册新版本。

建议总控下一步：
1. 不派发 `LIQUIDITY_BREAK_ONSET_V1` A1 target materialization。
2. 可冻结 A1 target support audit 计划文本，但状态必须为 blocked，
   future target access 必须保持零。
3. 若继续 alignment research，回到 M-state 定义，寻找具有独立
   persistence/novelty 语义的 onset，而不是当前高频 two-of-three
   threshold crossing。

提交信息：
- commit：`ae3048a3`
