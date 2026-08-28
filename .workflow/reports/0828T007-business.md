# 业务执行回报

执行线程：
- SKHYNIX Liquidity Break Onset A0 Contract 业务线程

任务ID：
- 0828T007

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0828T007.md`
- `docs/skhynix_binance_liquidity_break_onset_v1_a0_causal_anchor_contract_20260828.md`
- `.workflow/reports/0828T007-business.md`

action：
- 注册独立 hypothesis identifier `LIQUIDITY_BREAK_ONSET_V1`，明确不允许
  解释为 `OBI_REVERSAL_V1` 的 faster-alignment rescue。
- 将 primary alignment 冻结为原始 Binance 消息流中的
  `liquidity_break_onset_detected_at`。
- 冻结 event ordering 为
  `(local_receive_ts_ns,event_seq_in_file)`，相同 receive timestamp 保持
  文件顺序。
- 冻结 raw-message event-driven detector，不等待 100ms observation
  grid。
- 冻结 50ms primary window、L1-L5 `1/l` weights、60s causal robust
  baseline、500ms guard 和 20ms normalization checkpoints。
- 冻结三个可解释 pressure components：
  - vulnerable-side net depletion；
  - aggressive trade pressure；
  - whole-book flow pressure。
- 冻结 `z_star=3.0`、two-of-three coherence、aggregate score `6.0` 和
  direction conflict gap `0.5`。
- 冻结 `start_anchor` 为结构 predicate 首次成立的当前 message，不允许
  backdate、future dwell、maximum-pressure relocation 或 offline
  change-point replacement。
- 冻结 `end_anchor` 为 causal active-lock release，仅用于 duplicate
  suppression 和 support geometry，不作为 `n_liquidity_recovery`
  outcome。
- 冻结 outcome-blind control matching、adverse/recovery target stub、
  H0/H1 information families、A0 support gates 和 A1 timeliness stop
  gate。
- A1 timeliness gate 要求
  `pre_detection_adverse_fraction <= 0.30`，否则在 H0/H1 fitting 前停止
  当前版本。
- 明确 A0 不得读取 future midpoint/best-price target、不得拟合模型、
  不得采集新数据或访问任何 private/order surface。

verify：
- 对照现有 raw reconstruction、phase-alignment contracts、
  OBI reversal A0-A3 contracts/results 和 methodology kernel。
- Markdown fence count 为 `132`，配对完整。
- contract constants presence 检查通过。
- `git diff --check` 通过。
- contract line count 为 `1,118`。
- contract SHA256：
  `feea2f2b83361fe0fcb6d52ed60182af4c700db46b22d2e5f9a22db52673a8bc`。
- 使用 exact-path commit，未提交工作树已有无关 staged/untracked
  files。

done：
- A0 causal-anchor design contract 已在 Git commit `9d941862` 冻结。
- 当前冻结的是 design/authority boundary，不是 A0 detector execution
  result。
- 未实现或执行 detector，未产生 anchor ledger，未读取任何未来价格
  target，未拟合 H0/H1。
- 下一步只有在 QA 接受本 contract 后，才能派发独立 A0
  implementation/execution task。

blockers：
- 无合同起草与 Git 冻结阻塞。
- A0 execution 尚未授权。

commit：
- 9d941862

提交信息：
- docs: freeze liquidity break onset A0 contract
