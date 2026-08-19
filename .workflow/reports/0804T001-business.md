# 线程回报

执行线程：
- 业务线程-python/public-data-infra + cross-exchange-research

任务ID：
- 0804T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_collection_supervisor.py`
- `examples/hyperliquid/cross_exchange_alignment_acceptance.py`
- `examples/hyperliquid/test_cross_exchange_collection_supervisor.py`
- `examples/hyperliquid/test_cross_exchange_alignment_acceptance.py`
- `local_live_analysis/cross_exchange_collection_campaign_0804T001_skhynix_2h_continuous/`
- `local_live_analysis/skhynix_cross_exchange_research_0804T001/`
- `local_live_analysis/skhynix_cross_exchange_research_0804T001_old5h_replay/`

action：
- 增加 `--continuous-collection`，使完整 campaign 使用一个不中断的
  segment；增加 `--collection-only`，允许云端只采 raw data，延后全部
  timeline/R0/R1 计算。
- segment ready 之前的 Binance bookTicker 决策继续写入冻结 label audit，
  但标记为 warmup，不计入 depth-vs-bookTicker reconciliation missing。
- primary horizon label 改为目标时刻严格历史 as-of 的 Hyperliquid BBO
  阶梯状态。`source_age` 和 first-after-target 均保留为诊断字段，不允许
  future join 或跨 segment label。
- 修正 collection-only 的 status/abort/segment execution mode 诊断字段。
- 第一轮 QA 后增加 phase-specific runtime source archive：云端 collection、
  本地 postprocess、R0 builder、R1 builder 均保存实际源码副本和 SHA。
- 增加 `collection_control_plane_reconciliation.json`：不改写历史错误
  event，而是对原始 collection-only manifest、events 和
  `--skip-alignment` child command 做 SHA-bound reconciliation。
- 在 `c6in-winner` 使用 source-pinned runtime 完成一次授权的 SKHYNIX
  `7200s` public collection；云端没有运行 timeline、R0 或 R1。
- 采集结束后把全部 raw artifact 拉回本地，使用本机 conda 完成
  postprocess、R0 和 R1。

verify：
- Focused pytest：
  `PYTHONPATH=examples/hyperliquid /Users/liu/.local/conda/bin/python
  -m pytest -q examples/hyperliquid/test_cross_exchange_collection_supervisor.py
  examples/hyperliquid/test_cross_exchange_alignment_acceptance.py
  examples/hyperliquid/test_cross_exchange_l2_timeline.py
  examples/hyperliquid/test_cross_exchange_research_dataset.py`
  -> `71 passed`。
- Aug03 旧 5H R1 本地重放：
  `passes=true`、`reconciliation_pass=true`、八个 horizon 全 accepted、
  `future_join=0`，R0/source SHA 全部未变化。
- 新 2H 云端 unit：
  `hftbacktest-0804t001-skhynix-2h.service`，
  `ExecMainStatus=0`；采集时间为
  `2026-08-04 08:58:42-10:58:43 CST`。
- 拉回本地后六条 raw stream 的 SHA 全部匹配，所有 gzip 通过
  `gzip -t`。
- 第一轮 QA 后使用默认 hard stale gate 重新执行本地 postprocess；
  正式 timeline 记录
  `fast_stale_interval_policy.enabled=false`、所有 stale count 为 `0`。
- 本地 strict quality：
  `passes=true`、`failures=[]`、零 reconnect、零 degraded interval；
  双边 overlap `7200.094378508s`。
- 本地 timeline：
  `269,875` rows、`future_join=0`、`timestamp_regression=0`；
  fast-L2 age `p50=258.978727ms`、`p99=610.305026ms`、
  `max=1855.116715ms`，没有 stale interval。
- 本地 R0：
  `passes=true`、source hash unchanged；Binance hot
  `2,286,523` rows、Hyperliquid hot `136,237` rows、auxiliary
  `21,837` rows、mask `1` row。
- 本地 R1：
  `passes=true`、`reconciliation_pass=true`、`labels_pass=true`；
  `529,003` decision label rows，八个 horizon 覆盖率均为 `100%`，
  `future_join=0`、`timestamp_regression=0`、`cross_segment_label=0`。
- collection runtime 四个源码归档 SHA 与云端 manifest 一致；
  postprocess、R0、R1 runtime archive SHA 均与各自执行源码一致。
- 控制面 reconciliation `passes=true`，历史
  `collect_and_postprocess` 误标被保留并明确标记，最终 collection-only
  manifest 与 child `--skip-alignment` 事实闭合。
- Focused pytest 更新为 `72 passed`；`py_compile` 和
  `git diff --check` 通过。

done：
- 已修复 Aug03 数据暴露的 startup warmup reconciliation 假失败。
- 已修复把 BBO 未变价错误解释为 horizon label 缺失的问题。
- 已验证历史 5H 数据在新语义下通过 R1。
- 已完成新的连续 2H 采集、拉回和全本地 R0/R1 验证；本次修复有效。
- 已使用同一批 raw data 完成第一轮 QA 整改；未重新采集、未修改历史
  raw bytes 或历史控制事件。
- 仍只形成 alignment/research-input 资格，不构成 signal、arbitrage、
  exact fill、maker identity 或 maker PnL 结论。

blockers：
- 无

commit：
- 无

提交信息：
- 无
