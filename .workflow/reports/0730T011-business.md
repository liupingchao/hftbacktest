# 线程回报

执行线程：
- 业务线程-python/public-data-infra

任务ID：
- 0730T011

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0730T011.md`
- `.workflow/reports/0730T011-business.md`
- `examples/hyperliquid/cross_exchange_collection_supervisor.py`
- `examples/hyperliquid/test_cross_exchange_collection_supervisor.py`
- `docs/cross_exchange_collection_supervisor_l2_timeline.md`
- `local_live_analysis/cross_exchange_collection_campaign_0730T011_skhynix_4h_8x30m/`
- `local_live_analysis/0729T010_skhynix_4h_8x30m_postprocessed_0730T011.tar.gz`
- `local_live_analysis/0729T010_skhynix_4h_8x30m_postprocessed_0730T011.tar.gz.sha256`

action：
- 将 Hyperliquid tracks 分为：
  - core: `fast_market`, `standard_l2`
  - auxiliary: `asset_context`, `main_all_mids`,
    `target_dex_all_mids`
- Binance 和两个 Hyperliquid core L2 tracks 继续要求零重连。
- Auxiliary reconnect 仅在以下条件全部满足时降级通过：
  - reconnect、disconnect 和 connection-attempt 数量对账
  - 每次断线后所有 expected subscriptions 均重新 ACK
  - 每个 required channel 在 ACK 后恢复数据
  - track quality、raw row reconciliation、parse error、SHA 均通过
  - coverage、head/tail freshness 和 maximum arrival gap 均通过
- 从 raw local receipt timestamp 生成明确 degraded interval：
  - 起点是断线前最后一条 required-channel 数据
  - 终点是全部 ACK 后第一条恢复的 required-channel 数据
  - 策略使用 auxiliary features 时必须 exclude 或 mask
- 新增 `--postprocess-only`：
  - 禁止与 `--clean-output` 同时使用
  - 要求每个 segment 的 child results 和 sample manifests 完整
  - 不启动 collector，不访问交易所
  - 原始 collection source 保留在 `runtime_source.json`
  - 新 postprocess source 写入 `postprocess_runtime_source.json`
  - 旧 terminal control files 复制到 `postprocess_history/`
- 将修复部署到 c6in winner，保留原 supervisor：
  `cross_exchange_collection_supervisor.py.0729T010`.
- 对现有 T010 `8 x 1800s` raw campaign 原地执行 postprocess-only。
- 云端成功后生成归档、拉回本地并独立逐行验证。

verify：
- Local Conda env: `hftbacktest`.
- Focused supervisor tests:
  `32 passed in 0.32s`.
- Focused collection/timeline tests:
  `71 passed in 3.62s`.
- Full `examples/hyperliquid` regression:
  `1367 passed, 2 skipped in 65.60s`.
- `python -m py_compile`, CLI `--help` and `git diff --check` pass.
- Deployed supervisor SHA-256:
  `f2e123b2c865530ab4d86ddc0afc798a73dfffb96bb6758beae3ca799daf903f`.
- Preserved T010 supervisor SHA-256:
  `06f9d7806869dc575b16483481f91c6dbcc5ffab5f802156feb3cd1ab9c2a149`.
- Real T010 segment 2 recovery proof:
  - `asset_context`: `927.889576ms`
  - `main_all_mids`: `10029.488302ms`
  - all resubscription ACK and resumed-channel proofs present
- Remote postprocess:
  - unit: `hftbacktest-0730t011-postprocess.service`
  - result: success, exit status `0`
  - campaign state: `complete`
  - completed segments: `8/8`
  - timeline index rows: `8`
  - total common timeline rows: `556,861`
  - future joins: `0`
  - source timestamp regressions: `0`
  - source-age gates: `8/8`
  - raw SHA before/after postprocess: `48/48` unchanged
  - remote gzip: `56/56` passed
- Archive:
  - size: approximately `431M`
  - SHA-256:
    `df35a37aa421e92f2d4267ccb78cb5c31c365eaef54a0af5c78f9bd1910b71c4`
- Local independent verification:
  - archive SHA and tar/gzip structure pass
  - extracted size: approximately `434M`
  - files: `285`
  - gzip: `56/56`
  - raw SHA versus collection manifests: `48/48`
  - timeline SHA: `8/8`
  - CSV row counts: `8/8`, total `556,861`
  - every row has all three `source_local_ts_ns <= common_ts_ns`
  - every source age is nonnegative
  - local runtime source SHA matches remote postprocess provenance

done：
- 修复已完成并部署。
- 原始 4H 数据没有重新采集，`48/48` raw bytes 在 postprocess 前后
  保持一致。
- Segment 2 的两个辅助 reconnect 以可审计 degraded interval 保留；
  不再导致核心 L2 campaign 无条件失败。
- 八个 segment 的 common L2 timeline 已全部生成并通过。
- 完整数据和归档已拉回本地。

blockers：
- 无实现、运行或传输阻塞。
- 能力边界保持：
  - auxiliary degraded intervals 不能静默 forward-fill
  - segment 边界不声称 exact continuity
  - public L2 不提供 L3/L4 queue position 或 exact fill simulation

commit：
- 无

提交信息：
- 无
