# 线程回报

执行线程：
- 业务线程-python/public-data-infra + cross-exchange-research

任务ID：
- 0803T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 请独立复核 fast-L2 bounded stale-interval contract、remote postprocess、
  local pullback 完整性、R0 source-hash closure，以及 R1 fail 的准确归因。

files：
- `examples/hyperliquid/cross_exchange_l2_timeline.py`
- `examples/hyperliquid/cross_exchange_collection_supervisor.py`
- `examples/hyperliquid/cross_exchange_research_dataset.py`
- `examples/hyperliquid/cross_exchange_alignment_acceptance.py`
- `examples/hyperliquid/test_cross_exchange_l2_timeline.py`
- `examples/hyperliquid/test_cross_exchange_alignment_acceptance.py`
- local campaign:
  `local_live_analysis/cross_exchange_collection_campaign_0802T001_skhynix_5h_10x30m/`
- local R0/R1:
  `local_live_analysis/skhynix_cross_exchange_research_0803T001/`

action：
- 保留默认 fast-L2 source-age hard fail；仅新增显式 opt-in 的受限恢复：
  每个 stale interval 必须已恢复，单段和累计时长均受限，并写入 exact
  `fast_l2_staleness_interval` mask。
- Binance 与 Hyperliquid standard-L2 的 age/reconnect gate 未放宽。
- 部署修复到 c6in winner 的独立 source root
  `/home/admin/0803T001-hftbacktest`，对既有 raw campaign 执行
  postprocess-only。
- remote campaign 成功：10 个 segment，2 个 fast-L2 mask。
  `segment_0001` 为 `35.910592ms`，`segment_0003` 为 `71.580926ms`；
  两者均已恢复且低于每段/累计 `100ms` gate。
- 已将 `245M` campaign 拉回本机。campaign manifest `passes=true`；
  全部 `.gz` 文件通过 `gzip -t`。
- R0 成功：`638,026` timeline rows、`4,309,840` Binance hot rows、
  `279,578` Hyperliquid hot rows；70 个 source hashes unchanged；
  2 个 fast stale masks 被精确保留，未作无限 forward-fill。
- 修复 R1 对 mask boundary gap 的数值比较：`888.0954` 与
  `888.095400` 现在按 `1e-6ms` 数值容差比较；新增固定小数序列化回归测试。
- R1 provenance、exact masks、frozen labels、timestamp monotonicity 和
  future-decision join 均通过。R1 整体仍 fail closed，原因是独立的
  reconciliation/coverage gate，不是 fast-L2 age gate：
  `segment_0002` 至 `segment_0006` 的
  `binance_depth_vs_book_ticker` missing-asof count 超过冻结上限 `2`；
  全 campaign 无 primary horizon 达到每段 `95%` 有效标签覆盖率。

verify：
- remote:
  `hftbacktest-0803t001-skhynix-postprocess.service`
  `Result=success`, `ExecMainStatus=0`。
- local pullback:
  campaign `passes=true`，`segment_count=10`，gzip verification success。
- tests:
  `/Users/liu/.local/conda/envs/hftbacktest/bin/python -m pytest
  examples/hyperliquid/test_cross_exchange_alignment_acceptance.py
  examples/hyperliquid/test_cross_exchange_l2_timeline.py
  examples/hyperliquid/test_cross_exchange_collection_supervisor.py
  examples/hyperliquid/test_cross_exchange_research_dataset.py -q`
  -> `69 passed`。
- compile and diff:
  `py_compile` and `git diff --check` pass。
- R1:
  `exact_masks_pass=true`，`labels_pass=true`，
  `input_hashes_unchanged=true`，`future_decision_join_count=0`，
  `reconciliation_pass=false`，`passes=false`。

后段分段可用性：
- `segment_0004` 至 `segment_0010` 的 raw strict-quality 均为 pass，
  并且无 fast-L2 stale mask。
- `segment_0007` 至 `segment_0010` 的三项盘口 reconciliation 均通过，
  但 1000ms coverage 分别为 `92.3504%`、`93.5488%`、`92.6691%`、
  `92.9326%`，均低于冻结的 `95%`。
- 因此后段 raw/R0 可用于明确标注边界的诊断研究；当前没有任何后段可声称
  严格 R1-ready、可用于正式 signal/arbitrage conclusion。

done：
- fast-L2 age gate repair、remote postprocess、local integrity pullback 与 R0
  均完成。
- R1 已产出完整诊断包，但保持 fail closed，未将该批数据升级为 R1
  accepted evidence。

blockers：
- 需另行立项处理 Binance depth/bookTicker as-of 对账与 1s label coverage；
  不应通过放宽当前 frozen gate 来宣告本次 campaign 合格。

commit：
- 无

提交信息：
- 无
