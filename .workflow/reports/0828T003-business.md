# 业务执行回报

执行线程：
- SKHYNIX OBI Reversal Track A 方案更新业务线程

任务ID：
- 0828T003

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `docs/skhynix_binance_continuous_background_interpretable_m_state_competing_risk_track_a_plan_20260828.md`
- `.workflow/tasks/0828T003.md`
- `.workflow/reports/0828T003-business.md`

action：
- 将经验来源冻结为 `OBI_REVERSAL_V1`，并明确经验只用于选择假设和预期
  方向，不作为统计证据。
- 定义 equal-weight L1-L5 标准 OBI；L1、L1-L3、distance-weighted 和
  existing log-depth projection 仅为 robustness。
- 定义 old-state dwell、neutral crossing、opposite-state confirmation、
  quality/ambiguity 和 hysteresis 的 causal reversal state machine。
- 区分 `reversal_cross_at` 与 `reversal_detected_at`，将后者冻结为唯一
  primary decision/alignment time。
- 将 cross-to-detection 期间已发生的价格转移记为
  `pre_detection_transition`，禁止当作可捕捉成功。
- 构造同 current OBI band 的 reversal 与 non-reversal common risk set，
  检验相同 snapshot 下的 arrival-path increment。
- 定义 pressure-oriented `n_follow/n_fail` symmetric first-passage
  competing risks；无事件和数据边界进入 right censoring。
- 定义 H0=current OBI plus ordinary context，H1=H0 plus reversal indicator。
- 更新 proper scores、nulls、causal actionability、gates、failure
  classifications、artifacts 和 zero-target A0 first task。
- 将原 withdrawal/flow/refill composite M-state 降为独立 successor，
  禁止救回 OBI primary failure。

verify：
- `git diff --check`：通过。
- Markdown fenced-code markers 为 `114`，数量为偶数。
- 人工核对 `OBI_REVERSAL_V1`、OBI formula、cross/detect、matched
  controls、follow/fail、H0/H1、null 和 prospective boundaries。
- 人工核对 H0 必须包含 current OBI，H1 只增加 reversal path indicator。
- 人工核对 primary clock 从 `reversal_detected_at` 开始。
- 人工核对 generic composite M-state 不进入 OBI primary feature search。
- Git 提交使用 exact paths，未混入工作树已有无关 staged/untracked files。

done：
- OBI reversal 已作为 canonical Track A 计划的第一个、唯一 primary
  hypothesis。
- 已形成可执行的 current-OBI-matched path-dependence test。
- 历史数据最高结论仍为
  `historical_support_only_pending_prospective`；最终确认需要新的
  protocol-frozen prospective sessions。

blockers：
- 无文档任务阻塞。
- 科学结论仍存在 prospective 数据缺口；本任务未授权新采集。

commit：
- eeaec22f

提交信息：
- docs: add OBI reversal Track A hypothesis
