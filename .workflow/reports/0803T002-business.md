# 线程回报

执行线程：
- 业务线程-python/cross-exchange-research

任务ID：
- 0803T002

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 请独立复核 Jul30 方法冻结、Aug03 输入 provenance、diagnostic 边界传播、
  consumption ledger、双次确定性构建和统计结论。

files：
- `examples/hyperliquid/cross_exchange_liquidity_response_episodes.py`
- `examples/hyperliquid/cross_exchange_liquidity_response_baseline_v2.py`
- `examples/hyperliquid/cross_exchange_liquidity_response_regime_v2.py`
- `examples/hyperliquid/cross_exchange_liquidity_response_hierarchy_replay.py`
- `examples/hyperliquid/test_cross_exchange_liquidity_response_episodes.py`
- `examples/hyperliquid/test_cross_exchange_liquidity_response_hierarchy_replay.py`
- `local_live_analysis/skhynix_liquidity_response_0803T002/`
- `local_live_analysis/skhynix_liquidity_response_case_hierarchy_0803T002/`

action：
- 使用 Aug03 R0/event store 和失败的 R1 alignment manifest 构建独立
  diagnostic M1；没有修改或覆盖 Jul30 历史产物。
- discovery 固定为 `segment_0001-0003`，evaluation 固定为
  `segment_0004-0010`。
- 重放 ShockAtom、ShockCluster / ContinuousFlowEpisode、conditional
  baseline、Residual Motif / Prototype 和 TemporaryRegime v2 全管线。
- 每层发布 `diagnostic_status.json`，保留 source R1
  `passes=false`、`reconciliation_pass=false`，并固定
  `formal_eligible=false`。
- baseline consumption schema 按七个 evaluation segment 动态闭合；诊断
  reason 在 baseline、motif、regime 三层使用同一严格契约。
- replay runner 增加同输出目录独占文件锁，防止并发复跑互相清理固定
  `.tmp` 发布目录。

verify：
- 完整真实构建两次；55 个核心文件逐 SHA 一致，root manifest 的计数、
  boundary 和全部 stage manifest SHA 一致。
- 独立 validator：
  episode-v2、baseline-v2、post-selection、motif-v2、regime-v2 全部通过。
- tests：
  `/Users/liu/.local/conda/bin/python -m pytest -q
  examples/hyperliquid/test_cross_exchange_liquidity_response_episodes.py
  examples/hyperliquid/test_cross_exchange_liquidity_response_case_hierarchy.py
  examples/hyperliquid/test_cross_exchange_liquidity_response_baseline_v2.py
  examples/hyperliquid/test_cross_exchange_liquidity_response_motif_v2.py
  examples/hyperliquid/test_cross_exchange_liquidity_response_regime_v2.py
  examples/hyperliquid/test_cross_exchange_liquidity_response_hierarchy_replay.py`
  -> `70 passed`。
- `py_compile`、CLI `--help`、全量 `gzip -t` 和 `git diff --check` 通过。

done：
- M1：`127,622` candidates，`82,533` primary ShockAtoms；buy/sell
  `41,341/41,192`。最低 primary horizon coverage 为
  `90.4086265607%`。
- Episode v2：`48,790` ShockClusters，`24,040`
  ContinuousFlowEpisodes，`29,465` phases，`26` long-flow cases。
- Baseline v2：`6,486` discovery features / `305,418` predictions；
  `17,554` post-selection features / `411,406` predictions。
- Motif v2：`19` prototypes；199 次 full-pipeline surrogate；经验 p-value
  `0.98-1.0`，BH q-value 全为 `1.0`，全部 `not_supported`。
- Regime v2：`300` one-minute windows；999 次 primary surrogate；5 个
  spacing candidates 的 family-wise p-value 为
  `0.827/0.139/1.0/1.0/1.0`，global boundary-count p-value 为 `0.336`；
  `0` 个 data-driven boundary，发布的 `10` 个 interval 均为机械分段边界。
- 完成 Aug03 diagnostic hierarchy 发布，但没有形成正式 signal、
  arbitrage、exact fill、maker identity 或 maker PnL 结论。

blockers：
- Aug03 source R1 仍未通过 reconciliation/coverage gate。正式研究资格
  必须先由独立任务修复 alignment，而不是放宽本任务统计门禁。

commit：
- 无

提交信息：
- 无
