# QA 验收结果

执行线程：
- QA验收线程

任务ID：
- 0828T010

状态：
- 已通过

更新时间：
- 2026-08-28 14:00 CST

验收线程：
- QA验收线程

验收对象：
- SKHYNIX Safe Reentry After Flow Excursion A0 Contract 业务线程
- plan/task/report commit：`b0f629b2`
- report commit record：`bfb072d0`

验收范围：
- 验收 `SAFE_REENTRY_AFTER_FLOW_EXCURSION_V1` 是否构成独立、冻结、
  outcome-blind 的 A0 design contract。
- 验收 novelty、persistence、recovery、refractory 和 safe-reentry
  alignment 是否具有因果且无歧义的状态转换。
- 验收 support gates、controls、H0/H1 information boundary、downstream
  target stub 和 forbidden rescue 是否足以约束后续执行。

验收步骤：
1. 核对 predecessor failure 与新 hypothesis/version boundary。
2. 逐段核对 frozen constants、state machine、episode identity 和
   terminal semantics。
3. 核对 zero-outcome ledger、control construction 和 A0 passing gates。
4. 检查候选确认前反向切换、双边恢复和 refractory reset 边界。
5. 执行 Markdown fence parity、关键常量 presence 和
   `git diff --check`。

实际结果：
- 新 hypothesis 不通过提高旧 detector threshold 或筛选旧 anchors
  获得，版本边界成立。
- Predecessor 50ms crossing 仅作为 micro-pressure observation，不再直接
  产生 episode 或 alignment anchor。
- Novelty 要求完整 `1000ms` bilateral pre-quiet；缺失历史、reset、
  active process 或方向冲突均 fail closed。
- Persistence 在 `250ms` 内要求累计 `100ms` causal qualifying
  exposure；confirmation 发生在信息首次可知时，不 backdate。
- 候选确认前的 dominant direction switch 会拒绝候选；确认后的所有
  opposite crossings 均并入同一 episode。
- `RECOVERY_CANDIDATE` 先要求双向低压，再等待双边 depth 恢复；只有
  两侧均达到 baseline `0.80` 才进入 refractory。
- Refractory 要求连续 `1000ms` 低压和双边恢复；renewed pressure
  返回同一 episode 并重置时钟。
- Safe-reentry 只在 refractory completion 当前时点检查一次
  `spread>=2 ticks`、`abs(OBI)<=0.50` 和双边 depth，不等待未来有利
  状态。
- Controls 使用当前 state 匹配，并冻结 `5000ms` recent-excursion
  exclusion；不使用 future outcome exclusion。
- A0 gates 同时限制 event rate、跨日期支持、compression、common
  support 和 follow-up geometry；未通过只能停在 A0。
- Future midpoint、best price、contact、queue fill、markout、PnL 和
  H0/H1 fit 均明确禁止。
- 方案共 `1159` 行、`162` 个 Markdown fences，配对完整；关键常量
  presence 和 `git diff --check` 通过。

验收结论：
- 已通过
- 结论说明：
  - A0 design contract 已形成可执行且 fail-closed 的新研究版本；
    本次通过仅授权后续独立 A0 implementation/execution task，不代表
    excursion、safe reentry、spread capture 或收益假设成立。

通过项：
1. Novelty、persistence 与 refractory 三层语义相互独立且状态机闭合。
2. Alignment 点位于过程完成后的首次 causal safe-reentry
   availability，不使用 future-confirmed backdating。
3. Outcome boundary、controls、gates 和 forbidden rescue 完整。
4. Git 历史清楚区分 plan freeze、commit record 和 QA。

不通过项：
1. 无。

缺陷清单：
1. 无。

阻塞项：
- 无设计验收阻塞。
- Detector 尚未实现或执行，A0 empirical classification 尚不存在。

建议总控下一步：
1. 派发独立 A0 implementation/execution task，严格按 frozen constants
   生成 candidate、episode、recovery、refractory 和 safe-reentry
   ledgers。
2. A0 execution 只做 support/compression/control audit，不读取任何
   downstream outcome。
3. 仅当 classification 为
   `A0_safe_reentry_contract_supported` 时，才启动 A1 target support
   audit。

提交信息：
- commit：`c2e35d69`
