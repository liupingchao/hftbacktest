# 线程回报

执行线程：
- 业务线程-python/three-session-commonality-execution

任务ID：
- 0804T008

状态：
- 已通过

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_liquidity_response_episodes.py`
- `examples/hyperliquid/cross_exchange_liquidity_response_case_hierarchy.py`
- `examples/hyperliquid/cross_exchange_three_session_commonality.py`
- `examples/hyperliquid/test_cross_exchange_three_session_commonality.py`
- `examples/hyperliquid/cross_exchange_bbo_lag_surrogate.py`
- `examples/hyperliquid/test_cross_exchange_bbo_lag_surrogate.py`
- `examples/hyperliquid/cross_exchange_fast_l2_secondary.py`
- `examples/hyperliquid/test_cross_exchange_fast_l2_secondary.py`
- `examples/hyperliquid/cross_exchange_multitrack_lag_surrogate.py`
- `examples/hyperliquid/test_cross_exchange_multitrack_lag_surrogate.py`
- `local_live_analysis/skhynix_liquidity_response_0804T008/`
- `local_live_analysis/skhynix_liquidity_response_case_hierarchy_0804T008/`
- `local_live_analysis/skhynix_three_session_commonality_0804T008/`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 已构建 Aug04 缺失的 Atom/Episode/Hierarchy，并完成 Jul30 冻结的
  outcome-free `structural_family_v1` 向 Aug03/Aug04 的无重拟合转移。
- 已构建 classification-time labels、双向 BBO state/event、closure leg、
  formation driver、path survival、fee/latency/capacity 数据集。
- 已完成 2000-draw block bootstrap。
- 已把 Aug04 原生 BBO surrogate 输入同步到
  `amdserver:/home/molly/project/hftbacktest`，用 12 workers 完成 999 个
  deterministic lag surrogates，再拉回本地完成 SHA/row/ID 对账。
- 已依序完成 adjusted/unadjusted effect-retention、first-after-target、
  Hyperliquid fast-L2 secondary family 和完整 Hyperliquid multi-track
  surrogate 四项门禁。
- 已用 12 workers 在 `amdserver` 完成 999 个完整 surrogate，将
  `119,880` hypothesis rows 和 `6,993` 七轨 track-quality rows 拉回本地
  并执行 formal finalizer。
- 第一轮 QA 后升级为 v2：standard-L2、asset-context、main allMids、
  target-dex allMids 的完整 canonical state 在每个 `t+lag` 做 strict
  as-of 查询，并进入 `_auxiliary_state_ready` 资格重建。
- 已归档并绑定远端实际执行的 worker、fast-L2 secondary 与 commonality
  dependency 三份源码 SHA；finalizer 自身也单独归档。
- 已将旧 `directional_bbo_confirmation.md` 改写为 superseded，正式结论
  仅由 `directional_bbo_formal_gate_result.md` 和绑定 manifests 发布。
- 已将正式分层更新为 `43 C1 / 15 C2 / 2 C3`，并为每个未升级 hypothesis
  发布明确 blocker。

verify：
- 本机 `hftbacktest` conda env 已安装 `scikit-learn 1.9.0`、
  `scipy 1.18.0`、`joblib 1.5.3`、`threadpoolctl 3.6.0` 和
  `networkx 3.6.1`；`pip check` 通过。
- 最终联合回归：`164 passed in 98.18s`。
- 999 full-multitrack surrogate 共 `119,880` 行；primary/secondary 各
  `60` hypotheses x `999` slots，fit/quality/invalid failure 均为 `0`。
- 远端/本地正式输出 SHA 一致：
  `e38450dc240a44a12d6c7674b440ce3b9bd8317f546dc1f4d63d9354050c2c14`
  和
  `cd88a34e4e122c9adb11640f47330d2aae80c431dc43b9313c2863f298f1e776`。
- AMD 与 ARM surrogate 0 的非 beta 字段及七轨 quality 文件一致；
  beta 最大绝对尾差为 `6.995257706421398e-10`。
- 所有正式 gzip 通过 `gzip -t`。
- `commonality_manifest.json` 的 `77` 个文件记录逐文件 SHA/size 对账通过。
- finalize 重复执行后全目录 `78` 个文件 SHA 完全一致。
- `git diff --check` 通过。

done：
- 结构层：`46,866` assignments、`8` structural families；Jul30/Aug03/Aug04
  OOD 分别为 `5.57%/2.98%/3.89%`。
- BBO 机制层：`10,270` events、`51,350` leg rows、`8,749` positive paths；
  formation/leg identity failure 均为 `0`。
- effect-retention：`180` rows、`0` fit failures、`94` passes。
- first-after：`180` rows、最低 coverage `99.5079%`、`155` passes、
  `25` contradictions。
- fast-L2 secondary：`180` rows、`0` quality/fit failures，单独 BH family。
- full multi-track：`999` unique lags，每个 ID 精确 `7` 条轨道质量记录，
  order/no-wrap/query/checksum failure 为 `0`。
- `main_all_mids` 与 `target_dex_all_mids` 在极端负 lag 的订阅起始边界
  分别有累计 `3,701/3,608` 个缺失查询；全部计入 qualification
  exclusions 并被 `_auxiliary_state_ready` 排除。BBO 另有 `179,662`
  个超过 1 秒的查询被质量门禁排除。
- runtime source closure 为真，正式 run manifest 精确绑定输入 SHA、
  两个输出 SHA 和三份远端执行源码 SHA。
- 正式层级：`43 C1 / 15 C2 / 2 C3`。两个 C3 均为 `d_hb` level 对
  `1000/2000ms` survival，Aug04 `p/q` 分别为
  `0.007/0.042`、`0.005/0.0333333`。

blockers：
- 第一轮 QA 的三个缺陷已经修复：四条辅助状态轨实际查询、远端源码闭包
  和旧报告冲突均已关闭。
- `commonality_manifest.json` 仍保留
  `directional_bbo_formal_complete_structural_controls_pending`：
  directional-BBO 已 formal complete，但 broader structural commonality
  controls 不在本轮四门禁完成声明内。
- 第二轮独立 QA 待执行。

commit：
- 无

提交信息：
- 无
