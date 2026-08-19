# 线程回报

执行线程：
- 业务线程-python/public-data-infra + cross-exchange-replay

任务ID：
- 0805T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/hyperliquid_public_sample.py`
- `examples/hyperliquid/cross_exchange_collection_supervisor.py`
- `examples/hyperliquid/cross_exchange_l2_timeline.py`
- `examples/hyperliquid/cross_exchange_research_dataset.py`
- `examples/hyperliquid/cross_exchange_alignment_acceptance.py`
- 上述模块的 focused tests
- amdserver rebuilt campaign:
  `/home/molly/project/hftbacktest/local_live_analysis/0805T001_skhynix_4h_recovered_rebuild`
- amdserver R0:
  `/home/molly/project/hftbacktest/local_live_analysis/skhynix_cross_exchange_research_0805T001`
- amdserver R1:
  `/home/molly/project/hftbacktest/local_live_analysis/skhynix_cross_exchange_alignment_0805T001`

action：
- 保留 core-L2 reconnect 默认 hard failure；仅在显式 opt-in 且 transport
  marker、disconnect、全部 re-ACK、必要 channel 恢复、recovery snapshot、
  raw row/SHA 和 bounded interval 全闭合时接受 recovered reconnect。
- 将历史空 websocket frame 从 malformed parse error 中区分出来；非空
  malformed JSON 继续 fail closed。
- 为 Hyperliquid fast/standard L2 注入 reconnect boundary，断线时删除旧
  book state，直到新完整 L2 snapshot 才恢复 timeline 输出。
- 为 timeline 和 R0 hot events 发布 `connection_epoch_id`，并生成
  `core_l2_reconnect_interval` exact masks。
- R1 对每个 `[decision_ts, target_ts]` 做 mask 区间相交检查，排除所有
  intersecting horizons；跨 epoch label、future join 继续硬失败。
- QA 发现 gzip header 时间戳导致相同内容的压缩 SHA 不稳定，已将
  timeline、R0 和 R1 writer 改为 `mtime=0`、空 header filename 的
  deterministic gzip。
- 在 amdserver 从 2026-08-04 的 `0804T009` 原始 campaign 建立两个隔离
  副本并执行完整 postprocess/R0/R1；源 campaign 始终只读。

verify：
- 本机 focused regression：`90 passed in 0.56s`。
- 本机 `py_compile` 和 `git diff --check` 通过。
- amdserver 本地/远端四个核心 runtime source SHA 一致。
- 源 campaign `33` 个文件在全部重建前后 SHA 完全不变；六条 raw gzip
  与两个隔离副本逐文件一致。
- 两次 full postprocess 均通过；A/B timeline 都为 `501,495` 行，压缩
  SHA 都是
  `14161e17f331785a03cb86a5239146caf5f753f1d16357d4d2ffe3a16557ff86`。
- 两次 R0/R1 的 timeline、三个 R0 gzip、mask index、R1 labels、
  quality、coverage、age 和 top-of-book 文件全部字节级一致。
- 全部生成 gzip 通过 `gzip -t`。

done：
- strict quality：
  `raw_integrity_pass=true`、`continuous_exact_replay=false`、
  `segmented_replay_eligible=true`、
  `research_eligible_with_masks=true`。
- fast/standard L2 各有 `1` 次 recovered reconnect 和 `2` 个 connection
  epochs；source-age stale rows 全部为 `0`，旧状态跨 reconnect
  forward-fill 为 `false`。
- R0 通过：`3,907,048` Binance hot rows、`206,989` Hyperliquid hot
  rows、`38,618` auxiliary rows、`6` masks，其中
  `2` core reconnect、`3` auxiliary reconnect、`1` segment epoch。
- R1 通过：`738,058` decision rows、`737,928` eligible decisions、
  `130` reconnect-decision exclusions、`1,717` horizon-mask exclusions。
- `10/25/50/100/250/500/1000/2000ms` 八个 primary horizons coverage
  均为 `100%`；cross-epoch、future join 和 timestamp regression 均为 `0`。
- R1 decision-label gzip SHA 为
  `a4a1076cb7c9bfb5223149679f04d3791441dcd3da58820767bb38630b0e2cd7`。

blockers：
- 无。
- 能力边界仍然不包含 L3/L4 queue reconstruction、exact fill、
  executable arbitrage、maker PnL 或 live promotion。

commit：
- 无

提交信息：
- 无
