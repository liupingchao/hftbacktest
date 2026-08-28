# 业务执行回报

执行线程：
- SKHYNIX OBI Reversal Track A0 业务线程

任务ID：
- 0828T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0828T004.md`
- `docs/skhynix_binance_continuous_background_interpretable_m_state_competing_risk_track_a_plan_20260828.md`
- `examples/hyperliquid/skhynix_obi_reversal_track_a0.py`
- `examples/hyperliquid/test_skhynix_obi_reversal_track_a0.py`
- `local_live_analysis/skhynix_obi_reversal_track_a0_0828T004/`
- `.workflow/reports/0828T004-business.md`

action：
- 实现 zero-future-price-target A0 runner，复用已关闭身份的 29 个
  Binance public captures 和 100ms causal top-5 reconstruction cache。
- 验证 raw size/SHA、cache size/SHA、capture/date/role identity 和 depth-gap
  closure；本轮重新计算 raw hashes。
- 仅从 cache 读取 L1-L5 bid/ask quantity、timestamp、valid 和 provenance
  metadata，计算 equal-weight L1-L5 standard OBI。
- 冻结 `OBI_REVERSAL_V1` state machine：绝对 OBI threshold `0.50`、
  neutral band `0.10`、old-state dwell `1s`、opposite confirmation `1s`、
  cross-to-confirm timeout `10s`，primary alignment 为
  `reversal_detected_at`。
- 用 zero-target support trace 比较 `0.40/0.50/0.60`；选择最小满足
  event-rate/date-support/date-concentration gates 的 `0.50`。
- 生成 reversal entries、non-reversal control candidates、common risk-set
  overlap 和 unique no-reuse matched-pair diagnostic。
- 将 control gate 从错误的一对一 no-reuse matched coverage 修正为 hazard
  estimand 对应的 reusable common risk-set coverage；该修正在任何 future
  price target access 之前完成，并由回归测试锁定。
- 仅按 capture boundary 和 follow-up completeness 从
  `100ms` 到 `300s` 选择 horizon，冻结 `tau_max=120s`。
- 冻结 symmetric one-tick first-passage target contract、censoring、
  H0/H1、proper-score、null、uncertainty 和 numeric gates，但未生成
  follow/fail labels。
- 将 A0 实际结果、方法修正和下一阶段边界写回 canonical plan。

verify：
- `python -m pytest -q examples/hyperliquid/test_skhynix_obi_reversal_track_a0.py`：
  `5 passed`。
- `python -m py_compile examples/hyperliquid/skhynix_obi_reversal_track_a0.py examples/hyperliquid/test_skhynix_obi_reversal_track_a0.py`：通过。
- `python examples/hyperliquid/skhynix_obi_reversal_track_a0.py --verify-raw-hashes`：
  全量执行通过。
- 两个独立临时输出目录的 deterministic manifest 比较：21/21 artifact
  hashes 一致。
- 正式输出 `run_manifest.json` closure：21/21 artifact size/SHA 通过。
- outcome-access ledger：future price fields `[]`、follow/fail labels
  `false`、aligned future-price plots `false`。
- `git diff --check`：通过。
- Markdown fenced-code markers：`126`，数量为偶数。
- Git 使用 exact-path commit，未提交工作树已有无关 staged/untracked
  files。

done：
- 29 captures、9 dates、35.9172 hours 的 source/cache closure 完成。
- 23 个 captures 具有有效 OBI variation；6 个 low-variation captures
  保留 provenance 但不进入 state/control support。
- 冻结后识别 4,021 个 reversal entries，111.952/hour，最大单日占比
  0.2502。
- 生成 3,162 个 non-reversal control candidates。
- common-support reversals 为 4,005/4,021，覆盖率 0.9960，最低单日覆盖率
  0.8764。
- unique no-reuse matched pairs 为 2,801，覆盖率 0.6966，仅作为诊断。
- `tau_max=120s`：overall complete fraction 0.9724，最低单日 0.8276，
  9 个日期具有 complete pairs；`300s` 未通过。
- 所有 A0 gates 通过，classification 为
  `A0_support_and_tuple_frozen`。
- 下一阶段可以在新 formal task 中进入 A1 state materialization 和 A2
  first-passage target materialization。
- 当前不声称 OBI reversal 有预测力，也不声称历史结果是 prospective。

blockers：
- 无 A0 执行阻塞。
- OBI reversal 的统计增量仍未检验；future-price targets 尚未物化。
- 最终正结论仍需要 protocol-frozen prospective sessions。

commit：
- 663c14bd

提交信息：
- research: freeze OBI reversal A0 tuple
