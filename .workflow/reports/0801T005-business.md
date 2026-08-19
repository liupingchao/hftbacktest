执行线程：
- 业务线程-python/cross-exchange-research

任务ID：
- 0801T005

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_liquidity_response_case_hierarchy.py`
- `examples/hyperliquid/test_cross_exchange_liquidity_response_case_hierarchy.py`
- `local_live_analysis/skhynix_liquidity_response_case_hierarchy/`
- `.workflow/tasks/0801T005.md`
- `.workflow/reports/0801T005-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 新增 Goal 4 `--stage regime` builder。
- 使用 M1 timeline provenance 构建每段独立 non-overlapping `1min` context
  windows。
- 使用 discovery median/IQR context score 生成 internal boundary candidates。
- 对每段运行 `999` surrogate context sequences，使用 `5min` block permutation
  和 max-score p value 校准。
- 只发布 segment mechanical boundaries；所有 internal candidates 因未满足
  p<=`0.01` 留在 audit。
- 在 regime intervals 冻结后链接 episode membership 和 motif membership。
- 输出 `regime/` 全部 artifacts，并记录无 permanent ontology / signal /
  maker / fill / PnL 声称。

verify：
- `python -m pytest examples/hyperliquid/test_cross_exchange_liquidity_response_case_hierarchy.py -q`
  -> `10 passed`
- `python -m py_compile examples/hyperliquid/cross_exchange_liquidity_response_case_hierarchy.py examples/hyperliquid/test_cross_exchange_liquidity_response_case_hierarchy.py`
  -> pass
- 真实 Goal 4 builder：
  `python examples/hyperliquid/cross_exchange_liquidity_response_case_hierarchy.py --stage regime --m1-dir local_live_analysis/skhynix_liquidity_response_0730T017 --output-dir local_live_analysis/skhynix_liquidity_response_case_hierarchy --task-id 0801T005`
  -> pass。
- 独立 rescan：
  - context windows `240`
  - boundary audit rows `192`
  - published boundaries `8`
  - data-driven boundaries `0`
  - regime intervals `8`
  - episode-regime rows `12,677`
  - motif-by-regime rows `112`
  - surrogate rows `8`
  - independent errors `0`
- `git diff --check` 覆盖本轮代码、任务、QA 文档和总控文档 -> pass。

done：
- `regime/one_minute_context.csv.gz` row count `240`。
- `regime/regime_boundary_audit.csv` row count `192`。
- `regime/regime_boundaries.csv` row count `8`。
- `regime/regime_surrogate_summary.csv` row count `8`。
- `regime/regime_intervals.csv` row count `8`。
- `regime/episode_regime_membership.csv.gz` row count `12,677`。
- `regime/motif_by_regime.csv` row count `112`。
- `regime/regime_transition_summary.csv` row count `0`。
- 本轮未访问网络、AWS 或 SSH，未新增采集。

blockers：
- 无；等待独立 QA。

commit：
- 无

提交信息：
- 无
