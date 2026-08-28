# 业务执行回报

执行线程：
- SKHYNIX Interpretable Conditional Transition Track A 方案业务线程

任务ID：
- 0828T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `docs/skhynix_binance_continuous_background_interpretable_m_state_competing_risk_track_a_plan_20260828.md`
- `docs/skhynix_binance_continuous_background_recurrent_structural_excursions_track_a_plan_20260828.md`
- `.workflow/tasks/0828T002.md`
- `.workflow/reports/0828T002-business.md`

action：
- 将 motif-first、segmentation/DTW/clustering primary route 标记为
  superseded，保留为研究历史。
- 定义 hypothesis-driven primary route：continuous background、
  broad pressure landmark、interpretable M-state、mutually exclusive N
  transitions、competing-risk hazard 和 H0/H1 incremental test。
- 定义 `m1=pressure-side depth withdrawal`、
  `m2=aggressive-flow persistence`、`m3=replenishment deficit`，并使用
  `q_M=min(m1,m2,m3)` 表示可解释联合状态。
- 定义 `n_recovery`、`n_propagation`、`n_reversal` 三个互斥 first-event
  target；无事件、capture end、gap、reconnect、reset 和 quality failure
  进入 right censoring。
- 将 H0 收紧为 context 加 `m1/m2/m3` 加性主效应，H1 只新增 joint
  `q_M`，防止普通单变量自相关被误认为联合 pattern。
- 使用低参数 discrete-time multinomial competing-risk hazard，让
  elapsed-time baseline 表达数据决定的转移时间尺度。
- 定义 support-before-target、proper scores、blocked uncertainty、
  dependence-preserving null、historical no-refit transport、prospective
  confirmation、failure classifications 和 Track B unlock contract。
- 将 soft-DTW、k-medoids 和 shapelet 降为 primary H0/H1 gate 通过后的
  secondary diagnostics。

verify：
- `git diff --check`：通过。
- Markdown fenced-code markers 为 `96`，数量为偶数。
- 人工核对 M components 因果、方向统一、低维且不由 N outcome 调整。
- 人工核对 N states 互斥、first-event、tie/ambiguity 和 censoring 合同。
- 人工核对 H0 已包含 component main effects，H1 只增加 joint `q_M`。
- 人工核对 primary proper score 使用完整 `tau_max`，禁止挑选有利 horizon。
- 人工核对旧 motif-first 文档已明确 superseded。
- Git 提交使用 exact paths，未混入工作树已有无关 staged/untracked files。

done：
- 已形成 2026-08-28 interpretable conditional transition Track A 正式方案。
- 已明确现有历史数据最多达到
  `historical_support_only_pending_prospective`，最终确认需要新的
  protocol-frozen prospective Binance public sessions。
- 下一执行任务被限制为 A0 support/tuple freeze，zero N-target access。

blockers：
- 无文档任务阻塞。
- 最终正结构结论存在 prospective 数据缺口；本任务未授权新采集。

commit：
- 75daafa8

提交信息：
- docs: add interpretable transition Track A plan
