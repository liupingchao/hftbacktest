# 线程回报

执行线程：
- 业务线程-python/cross-exchange-research

任务ID：
- 0730T014

状态：
- 待验收

是否进行QA验收：
- 是

files：
- `.workflow/tasks/0730T014.md`
- `.workflow/reports/0730T014-business.md`
- `examples/hyperliquid/cross_exchange_research_dataset.py`
- `examples/hyperliquid/test_cross_exchange_research_dataset.py`
- `docs/skhynix_cross_exchange_research_dataset.md`
- `local_live_analysis/skhynix_cross_exchange_research_0730T013/`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- `--clean-output` 改为先在 sibling temporary directory 完整构建；
  发布时使用 backup/rename/rollback，失败不会删除旧产物。
- Timeline index、manifest 和真实 gzip CSV 闭环校验：
  campaign/profile/segment、continuity、SHA、row count、first/last timestamp；
  同时逐行校验 CSV 内 campaign/segment/profile identity。
- 按相邻段首尾时间独立重算 `previous_segment_gap_ms`。
- Timeline `source_raw` SHA 与当前 Binance、Hyperliquid fast L2、
  standard L2 source 闭环。
- Binance depth/bookTicker/trade 全部校验 symbol，并执行精确
  snapshot/event-type count 门禁。
- Hyperliquid fast/standard/auxiliary 执行 collection/bundle SHA、coin/dex、
  精确 channel/control set、raw row count、parse-error reconciliation。
- Main allMids 强制无 named dex；target-dex allMids 强制目标 dex。
- Standard L2 即使不写入 hot sidecar，也完整扫描 coin、levels、时间序和
  channel/count。

verify：
- Local Conda env:
  `/Users/liu/.local/conda/envs/hftbacktest`.
- Failure-injection tests: `18 passed`.
- Dataset + timeline + supervisor regression: `59 passed in 0.32s`.
- `py_compile`, CLI `--help` and `git diff --check` pass.
- Failed clean rebuild regression confirms prior manifest bytes remain unchanged。
- Real eight-segment rebuild:
  - segments: `8`
  - output size: `298M`
  - files: `34`
  - Binance hot rows: `10,744,733`
  - Hyperliquid hot rows: `541,122`
  - Hyperliquid exploded trade items: `398,224`
  - Hyperliquid auxiliary rows: `49,169`
  - common timeline rows: `556,861`
  - mask rows: `10`
  - source hash count: `56`

done：
- T013 QA 的三个 P2 builder contract 缺陷已修复。
- T014 首轮 QA 补充发现的 timeline row identity 和 main-allMids dex
  两个 P2 已修复并加入 failure injection。
- 真实 R0 package 已在旧产物保留条件下成功重建和发布。
- T014 进入 `待验收`；R1 继续等待独立 QA。

boundary：
- 仅使用现有本地数据。
- 未访问 network、AWS、SSH 或交易端点。
- 未新增采集、未拟合信号。
- 后续新增采集仍需要独立任务、用户明确授权和用户确认的活跃交易时段。

commit：
- 无
