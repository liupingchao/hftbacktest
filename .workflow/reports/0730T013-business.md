# 线程回报

执行线程：
- 业务线程-python/cross-exchange-research

任务ID：
- 0730T013

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0730T013.md`
- `.workflow/reports/0730T013-business.md`
- `examples/hyperliquid/cross_exchange_research_dataset.py`
- `examples/hyperliquid/test_cross_exchange_research_dataset.py`
- `docs/skhynix_cross_exchange_research_dataset.md`
- `local_live_analysis/skhynix_cross_exchange_research_0730T013/`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 新增 local-only R0 builder：
  `cross_exchange_research_dataset.py`.
- 对 campaign、timeline index、strict quality、timeline SHA/row count 和
  六类 raw source SHA 执行 fail-closed validation。
- Common L2 timeline 保持为 replayed L2 state source，不复制其
  `556,861 x 337` 字段。
- 生成八段：
  - `binance_hot_events.csv.gz`
  - `hyperliquid_hot_events.csv.gz`
  - `hyperliquid_auxiliary_events.csv.gz`
  - `segment_event_store_manifest.json`
- Binance sidecar 标准化 bookTicker/trade，并保留 update id、BBO、trade id、
  buyer-maker flag 和 aggressor side。
- Hyperliquid sidecar 标准化 BBO 并展开 batched trades，保留 raw side、
  trade id、hash 和 users。
- Auxiliary sidecar 使用三路 local-time heap merge，保留 full compact
  payload JSON 和 degraded interval flags。
- 生成 `segment_and_mask_index.csv`：
  - `8` 个 segment epoch
  - `2` 个 auxiliary degraded intervals
- 发布前后重新计算所有 source SHA，确认输入未变化。

real-run repairs：
- 第一轮真实构建发现 auxiliary raw 包含 `{"channel":"pong"}` control
  rows；修复为计数但不解析为 data event，并增加 synthetic regression。
- 第二轮独立扫描发现三个 auxiliary tracks 最初按 track 拼接，跨 track
  local timestamp 会回退；修复为 heap merge，并增加统一时间轴回归测试。
- 修复后删除并重建原生成目录；最终产物没有 timestamp regression。

verify：
- Local Conda env:
  `/Users/liu/.local/conda/envs/hftbacktest`.
- Synthetic focused tests:
  `5 passed`.
- Dataset + timeline + supervisor focused regression:
  `46 passed in 0.27s`.
- `py_compile`, CLI `--help` and `git diff --check` pass.
- Real build:
  - duration: approximately `66.21s`
  - output size: approximately `298M`
  - files: `34`
  - segments: `8`
- Independent output scan:
  - Binance bookTicker: `5,626,969`
  - Binance trades: `5,117,764`
  - Binance hot total: `10,744,733`
  - Hyperliquid BBO: `142,898`
  - Hyperliquid exploded trade items: `398,224`
  - Hyperliquid hot total: `541,122`
  - Hyperliquid activeAssetCtx: `14,101`
  - Hyperliquid candle: `29,355`
  - Hyperliquid allMids: `5,713`
  - Hyperliquid auxiliary total: `49,169`
  - common L2 timeline rows referenced: `556,861`
  - mask rows: `10`
  - output SHA/row-count/timestamp/symbol/coin problems: `0`
- Degraded output rows:
  - segment 2 asset_context: `5`
  - segment 2 main_all_mids: `2`
- Independent source rehash:
  - checked: `56`
  - changed/mismatched: `0`

done：
- R0 research event store 已完整生成并独立复核。
- Original raw 和 common timeline 均保持 byte-identical。
- Segment boundary、source provenance 和 degraded masks 均显式可消费。
- R0 进入`待验收`；R1 尚未开始。

blockers：
- 无实现或本地数据构建阻塞。
- T011 postprocess 独立 QA 仍为 pending，并保留在
  `research_input_manifest.json` source status。
- 新增数据采集未发生，未来仍需要用户明确授权和活跃交易时段确认。

commit：
- 无

提交信息：
- 无
