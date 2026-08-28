# 业务执行回报

执行线程：
- SKHYNIX OBI Reversal Track A1/A2 业务线程

任务ID：
- 0828T005

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0828T005.md`
- `docs/skhynix_binance_continuous_background_interpretable_m_state_competing_risk_track_a_plan_20260828.md`
- `examples/hyperliquid/skhynix_obi_reversal_track_a1_a2.py`
- `examples/hyperliquid/test_skhynix_obi_reversal_track_a1_a2.py`
- `local_live_analysis/skhynix_obi_reversal_track_a1_a2_0828T005/`
- `.workflow/reports/0828T005-business.md`

action：
- 验证 0828T004 A0 的 21 个 artifacts、classification、state machine、
  control 和 target contracts 的 size/SHA closure。
- 验证 29 个 0827T004 feature caches 的 identity、schema 和 SHA。
- A1 复现 4,021 reversals 和 3,162 controls，物化 exact
  decision-time H0 context。
- 按 frozen date/side/absolute-OBI-bin common support 生成 7,164 个
  primary entries，其中 4,005 reversals、3,159 controls。
- A2 从 decision 后下一 100ms grid 开始累加
  `midpoint_delta_ticks`，生成 frozen symmetric one-tick follow/fail
  first-passage targets。
- 仅保存 first transition type、cause code、event time 和 censoring；
  未保存 post-hit displacement、return、markout 或后续路径。
- 单独物化 cross-to-detection pre-detection transition。
- 执行 target variation、date support、date concentration、censoring 和
  ambiguity gates。
- 将 A1/A2 结果、temporal-resolution warning、confirmation-delay
  warning 和 A3 边界写回 canonical plan。

verify：
- `python -m pytest -q examples/hyperliquid/test_skhynix_obi_reversal_track_a0.py examples/hyperliquid/test_skhynix_obi_reversal_track_a1_a2.py`：
  `10 passed`。
- Python compile 和 `git diff --check`：通过。
- 两个独立临时输出的 deterministic manifest：15/15 artifacts 完全一致。
- 正式 run manifest：15/15 artifact size/SHA closure 通过。
- target ledger schema 人工与自动核对：不存在 exit displacement、
  markout、return 或 PnL 字段。
- outcome-access ledger：A1 future fields `[]`；A2 future fields 仅
  `midpoint_delta_ticks`；H0/H1 未拟合。
- Markdown fenced-code markers：`146`，数量为偶数。
- Git 使用 exact-path commit，未提交已有无关 staged/untracked files。

done：
- A1/A2 classification 为
  `A1_A2_state_and_targets_materialized`，全部 A1/A2 gates 通过。
- 7,164 primary entries 中 follow 4,247、fail 2,917，无 censoring。
- Follow/fail 均覆盖 9 日期；最大单日占比分别 0.3421 和 0.2756。
- Reversal entries 为 follow 2,345、fail 1,660；controls 为 follow
  1,902、fail 1,257。
- Follow event time p50/p90 为 300ms/1,500ms；fail 为
  400ms/1,600ms。
- 24.46% entries 在第一格 100ms 命中，66.02% 在 500ms 内命中，
  30s 内全部命中，记录
  `one_tick_barrier_near_100ms_grid_resolution`。
- 4,005 primary reversals 中 2,986 pre-detection follow、716
  pre-detection fail、303 none；任一 pre-detection hit 比例 0.9243，
  记录 `high_pre_detection_transition_fraction`。
- 未执行 H0/H1 增量检验，不允许 predictive-value claim。
- A3 已获得执行资格，但必须使用 frozen target，并携带两个 warnings。

blockers：
- 无 A1/A2 materialization 阻塞。
- 一跳目标接近 100ms grid resolution，grid 内双 barrier 路径不可观测。
- confirmation 前已有价格转移的比例很高，构成重大 actionability 风险。
- OBI reversal 的 path-dependence increment 仍须 A3 检验。

过程说明：
- 首次 direct CLI 在读取 target 前因仓库模块搜索路径报
  `ModuleNotFoundError`；补充 package/direct-script 双路径 import 后通过。
- 初版内部 target row 曾包含 first-hit exit displacement；artifact 审查
  时发现它超出允许保留的 `J/T/censoring` 边界，已在最终运行和提交前
  删除，并增加 post-first-hit mutation 测试。

commit：
- f0361388

提交信息：
- research: materialize OBI reversal A1 A2 targets
