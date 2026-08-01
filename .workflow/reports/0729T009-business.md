# 线程回报

执行线程：
- 业务线程-python/public-data-infra

任务ID：
- 0729T009

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `.workflow/tasks/0729T009.md`
- `.workflow/reports/0729T009-business.md`
- `docs/cross_exchange_collection_supervisor_l2_timeline.md`
- `examples/hyperliquid/cross_exchange_symbol_registry.py`
- `examples/hyperliquid/synchronized_public_collection.py`
- `examples/hyperliquid/cross_exchange_collection_supervisor.py`
- `examples/hyperliquid/cross_exchange_l2_timeline.py`
- `examples/hyperliquid/test_cross_exchange_collection_supervisor.py`
- `examples/hyperliquid/test_cross_exchange_l2_timeline.py`
- `local_live_analysis/cross_exchange_collection_campaign_0729T009_4profiles_2x30s_final_v5/`
- `local_live_analysis/cross_exchange_collection_campaign_0729T009_4profiles_2x30s_final_v5.tar.gz`

action：
- 新增四个 public-data symbol profiles：
  - `btc`: Binance `BTCUSDT` / Hyperliquid `BTC`
  - `eth`: Binance `ETHUSDT` / Hyperliquid `ETH`
  - `skhynix`: Binance `SKHYNIXUSDT` / Hyperliquid `xyz:SKHX`
  - `mu`: Binance `MUUSDT` / Hyperliquid `xyz:MU`
- 实现 multi-profile campaign supervisor：
  - 同 segment 内每个 profile 一个 synchronized collector 子进程并发运行。
  - 所有 segment 先背靠背采集，再统一执行质量检查和 timeline 展开，
    避免 CSV 后处理形成长采集空窗。
  - 子进程独立 process group、timeout、TERM/KILL/wait/reap。
  - wrapper 退出后检查进程组；遗留后代执行 TERM/KILL 并失败关闭。
  - 原子 `run_status.json`、`heartbeat.json` 和 event log。
  - terminal heartbeat 明确写入 `heartbeat_stopped`。
  - 成功或失败 segment 都持久化 child pid、command、returncode、log
    和 process-group cleanup 结果。
  - 任一 child、质量门禁或 timeline 失败即 campaign fail closed。
  - 非空 output 默认拒绝；仅显式 `--clean-output` 可清理重跑。
- 实现严格 profile/segment 质量门禁：
  - Binance snapshot bridge、`depth_replay_ready`、continuity gap `0`。
  - Binance depth/trade/bookTicker 非空、duration、overlap、零 reconnect。
  - Hyperliquid research-max/all-tracks、fast/standard shape、ACK/row
    reconciliation、duration、overlap、零 reconnect。
  - main DEX (`BTC/ETH`) 要求 4 轨；named DEX (`xyz:*`) 额外要求
    target-dex allMids。
  - Binance/Hyperliquid 必需 channel 的覆盖率、尾部新鲜度和最大
    arrival gap。
  - 全部 raw SHA 与 manifest 对账。
- 实现 common L2 timeline：
  - Binance snapshot + depth diff 完整应用，数量 `0` 删除档位。
  - timeline 自行验证 snapshot 到首个 diff bridge，后续验证
    `pu == previous u`。
  - Hyperliquid fast 和 standard L2 分别按完整快照恢复。
  - 三轨按同机 `time.time_ns()` 做 union-event + strict as-of。
  - 每行包含 common/trigger identity、各 source local/exchange ts、age、
    top-20 px/qty 和 Hyperliquid `n`。
  - future join、source timestamp regression、replay gap 均失败关闭。
  - Binance/HL fast/HL standard source age 默认上限为 `2s/2s/15s`。
  - gzip 先写临时路径，成功后原子替换；失败删除旧终态和半文件。
- 每个 campaign 封存 supervisor、collector、timeline 和 registry 的
  runtime source SHA。
- `timeline_index.csv` 显式记录 segment 边界和相邻 segment gap，不声称
  cross-segment exact continuity。

verify：
- Focused regression：
  - `63 passed in 3.66s`
  - `python -m py_compile` 通过
  - 两个新 CLI `--help` 通过
  - `git diff --check` 通过
- 故障注入覆盖：
  - child nonzero、peer reap 和 failed-wrapper orphan descendant cleanup
  - segment timeout 和 reap
  - success/failure child result persistence
  - all-segments-before-postprocessing order
  - missing manifest/raw
  - snapshot bridge failure和后续 continuity gap
  - reconnect、failed research bundle 和 raw SHA mismatch
  - stale coverage/tail/arrival gap
  - unbounded forward-fill/source-age failure
  - timeline failure abort 和 local timestamp regression
  - negative/out-of-range gate parameters
  - stale output/timeline artifact rejection
- Full Hyperliquid suite：
  - 本机 Conda env: `hftbacktest`
  - Python `3.12.13` / NumPy `2.2.6` / Numba `0.65.0`
  - 排除一个既有 deadline timing flaky：
    `1357 passed, 2 skipped, 1 deselected in 62.41s`
  - 该 timing test 隔离重跑：`1 passed in 3.77s`
  - 未排除时两次仅该断言失败：
    `0.260034/0.260037s > 0.260000s`，超出 `34/37us`
- c6in winner 最终源码 `2 x 30s` 四 profile canary：
  - host: `i-0a962e47210528526 / c6in.xlarge`
  - campaign state: `complete`
  - terminal heartbeat: `heartbeat_stopped`
  - active children: `0`
  - 两个 segment、八个 profile-segment strict quality 均通过
  - total common L2 rows: `8714`
  - segment 2 common timeline gap:
    - BTC `1236.574798ms`
    - ETH `1223.738625ms`
    - SKHYNIX `1222.624943ms`
    - MU `1323.322063ms`
  - venue collection restart gap:
    - Binance `0.836-1.106s`
    - Hyperliquid `0.862-1.064s`
  - 所有 timeline `future_join_count=0`
  - 所有 source timestamp regression `0`
  - 所有 source-age gates `true`
  - 所有 required-channel coverage/tail/arrival-gap gates `true`
  - 所有 child/research tracks reconnect `0`
  - 所有 child post-exit process groups clean
- Runtime source SHA：
  - collector:
    `70faaff444b582d5068050f2c7dcdcd72de2935db50603753367a39c0029fbc4`
  - supervisor:
    `06f9d7806869dc575b16483481f91c6dbcc5ffab5f802156feb3cd1ab9c2a149`
  - registry:
    `b8237f8439c54dd8d6f44ac13a879bcec2735721e0c65c94f67c656f59311f69`
  - timeline:
    `c1651853ac9d38f022e87edf2f999971998a6941fe76855801ea15591434d41f`
- Pullback 独立验证：
  - 八份 gzip/CSV 可读。
  - CSV row count 与 manifest 一致。
  - 每行三个 `source_local_ts_ns <= common_ts_ns`。
  - 所有 source age 非负且 source-age gate 通过。
  - 八个 timeline SHA 与 manifest 一致。
  - `timeline_index.csv` 有标准表头，`csv.DictReader` 精确解析 `8` 行。
  - 本地源码 SHA 与远端 runtime source SHA 一致。
  - tar SHA:
    `8f512ab7c0353c5bf9e82aaf70eca9148a7beefcc1975ebecd3e9838aab0ea9b`

done：
- 已具备 BTC、ETH、SKHYNIX、MU 多标的双交易所分段 public-data
  campaign 采集能力。
- 已能在同机本地接收时间框架下恢复 Binance L2、Hyperliquid fast L2
  和 Hyperliquid standard L2，并生成可直接用于 lead-lag 特征研究的
  strict as-of 数据集。
- 最终 canary 中 Hyperliquid fast state age p50 约 `250ms`，
  standard state age p50 约 `2.5s`；两轨信息维度不同，必须独立保留。
- 支持 8H 配置，例如总时长 `28800s`、segment `1800s`；本任务没有
  用付费实例实际等待完整 8H。

blockers：
- 无代码或测试阻塞。
- 测试残余：
  - 一个与本任务无关的既有 deadline timing 断言在 full-suite 负载下
    稳定超阈值 `34-37us`，隔离重跑通过；本任务未修改该 live watcher。
- 残余能力边界：
  - segment 间从新 snapshot 启动，实测 restart gap 约 `0.75-1.42s`，
    不声称 exact continuity。
  - public L2 不提供 L3/L4 queue position。
  - 当前只支持 L2/as-of 研究，不声称 exact fill simulation。
  - 8H 稳态仍需后续正式长窗口运行验收。

commit：
- 无

提交信息：
- 无
