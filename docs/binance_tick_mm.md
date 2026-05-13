# Binance Tick MM 工作流分析

本文梳理 `examples/binance_tick_mm` 这套工作的结构、逻辑和当前对齐进展。这个目录的目标不是单纯跑一个策略示例，而是建立一条闭环：

```text
AWS 东京实盘运行
  -> 写出 audit_live.csv 和原始行情 gzip
  -> 本地生成订单延迟模型、转换行情、构造同窗口回放
  -> 运行 hftbacktest 回测
  -> 对比 audit_bt.csv 与 audit_live.csv
  -> 校准延迟、cadence、队列/成交、风控和策略参数
  -> 再做策略优化和 walk-forward / sweep
```

核心目标是：本地回测能够复现实盘发生过的市场、延迟、决策节奏、订单行为和仓位路径。只有这个闭环足够可信，后续参数优化才有意义。

## 1. 目录角色

`examples/binance_tick_mm` 是一个围绕 Binance USD-M Futures BTCUSDT 的 tick 级做市实验工程。

```text
examples/binance_tick_mm
├── README.md                    # 使用说明和部署/校准命令
├── config.example.toml           # 通用配置模板
├── strategy_core.py              # 回测和实盘共用的策略核心
├── backtest_tick_mm.py           # 本地回测入口
├── live_tick_mm.py               # AWS live bot 入口
├── pipeline.py                   # Tardis 历史数据 -> hftbacktest NPZ/manifest
├── pipeline_live_raw.py          # 实盘 collector gzip -> hftbacktest NPZ/manifest
├── latency_from_audit.py         # audit_live -> IntpOrderLatency NPZ
├── compare_audit.py              # audit_bt vs audit_live 对齐报告
├── backtest_metrics.py           # summary/daily metrics 与 audit 输出策略
├── run_env_test.py               # mac/amdserver 单日测试入口
├── walk_forward.py               # 滚动 train/test 回测
├── sweep_backtest.py             # 参数网格并行扫描
├── validate_audit.py             # audit schema 与公式口径校验
├── hash_audit.py                 # audit 文件确定性哈希
├── plot_audit.py                 # returns/position 图
├── deploy/
│   ├── binancefutures.toml       # connector 配置模板
│   └── run_live.sh               # tmux 启动 collector + connector + bot
└── test_*.py                     # 针对 manifest、cadence、live raw 转换等的单元测试
```

相关实盘分析产物集中在：

```text
local_live_analysis/live_btcusdt_1777342116/
```

这里保存了 2026-04-28 这次 live run 的 audit、原始行情、转换后的 NPZ、回测结果、对齐报告和总结。

## 2. 关键设计

### 2.1 回测和实盘共用策略核心

`strategy_core.py` 是这套工作的核心抽象。它把策略中与运行环境无关的逻辑抽出来，供 `backtest_tick_mm.py` 和 `live_tick_mm.py` 同时调用。

主要组件：

- `EwmaSigma`: 用 mid price log return 估计短周期波动。
- `TokenBucket`: 模拟或执行 API 速率限制。
- `GreekOracle`: 从 CSV 或当前仓位生成 Greeks 信号。
- `compute_top5_size`: 读取 ROI 盘口 top 5 bid/ask 数量。
- `impact_cost`: 分段线性冲击成本。
- `collect_working_orders`: 从 hftbacktest orders map 中提取当前 buy/sell 工作单，并发现多余订单。
- `decide_actions`: 根据目标报价和当前工作单生成 cancel/submit 动作。
- `build_audit_row`: 统一构造 audit 行。

策略核心公式大致是：

```text
mid = (best_bid + best_ask) / 2
sigma = EWMA(log(mid_t / mid_{t-1}))
bid_size, ask_size = top5 size

greek_adjustment =
  w_delta * delta + w_gamma * gamma + w_vega * vega + w_theta * theta

fair =
  mid
  + w_imb * (bid_size - ask_size)
  + w_spread * spread
  + w_vol * sigma
  + greek_adjustment

reservation = fair - k_inv * position

half_spread =
  base_spread
  + k_vol * sigma
  + k_pos * abs(position)
  + impact_cost(order_notional)
  + min(0.05, sigma * 0.1)
```

目标报价被约束在当前 best bid/ask 附近：

```text
target_bid = clamp(reservation - half_spread, best_bid * 0.999, best_bid)
target_ask = clamp(reservation + half_spread, best_ask, best_ask * 1.001)
```

`decide_actions` 保持一个重要不变量：正常情况下每边最多一个工作单。发现多余订单时先撤多余单；如果目标价格偏离现有挂单超过 1 tick，则先撤再挂；触发仓位上限时只保留减仓方向。

### 2.2 回测配置

`backtest_tick_mm.py` 当前使用的 hftbacktest 组合是：

```text
ROIVectorMarketDepthBacktest
  + BacktestAsset
  + LinearAsset
  + IntpOrderLatency
  + PowerProbQueueModel3(n)
  + NoPartialFillExchange
  + TradingValueFeeModel
```

配置中的主要校准点：

- `[latency] order_latency_npz`: 由实盘 audit 生成的 `IntpOrderLatency` 数据。
- `[latency] latency_guard_ms`: 延迟保护阈值，当前常用 5ms。
- `[queue] power_prob_n`: 队列概率模型参数，主要影响 fill rate。
- `[api_limit]`: token bucket 与最小 API 间隔。
- `[risk]`: 仓位、单笔名义金额、库存惩罚、基础半价差。
- `[fair]` / `[greeks]`: 公允价信号。
- `[audit]`: `full/actions_only/sampled/off`，决定 audit 写出成本。
- `[summary]`: 大规模回测时输出 summary JSON 和 daily CSV。
- `[backtest_cadence]`: 控制策略决策节奏，支持 `fixed_interval` 和 `audit_replay`。

`backtest_tick_mm.py` 还支持：

- 多日 manifest 连续回测。
- 初始 snapshot 应用。
- `first_5m/first_2h/first_6h/full_day` 窗口。
- 用绝对 `ts_local` 切片同一实盘窗口：

```bash
python examples/binance_tick_mm/backtest_tick_mm.py \
  --config /path/to/config.toml \
  --manifest /path/to/manifest.json \
  --window full_day \
  --slice-ts-local-start <live_first_ts_local> \
  --slice-ts-local-end <live_last_ts_local>
```

这对实盘复现很重要：如果只用 `first_2h` 这种相对窗口，回测时间段可能和实盘时间段错开。

### 2.3 实盘入口

`live_tick_mm.py` 使用：

```text
LiveInstrument
ROIVectorMarketDepthLiveBot
Rust connector IPC
```

运行时由 connector 推送真实行情、订单回报和仓位，策略循环用与回测相同的接口：

- `hbt.wait_next_feed(...)`
- `hbt.depth(0)`
- `hbt.orders(0)`
- `hbt.position(0)`
- `hbt.submit_buy_order(...)`
- `hbt.submit_sell_order(...)`
- `hbt.cancel(...)`

实盘和回测的主要差异：

- live 的事件来源是真实 connector，而不是 `Reader<Event>`。
- live 的 latency guard 当前只用 `feed_latency_ns` 作为前瞻保护信号。
- backtest 的 latency guard 同样只用 `feed_latency_ns` 作为前瞻保护信号，`predicted_entry_ns` 只作为诊断字段。
- live 退出时会尝试撤销所有工作单并关闭 bot。

`deploy/run_live.sh` 会在 tmux session `hft_live` 中启动三块：

```text
Pane 0: collector  # 录制 Binance Futures 行情
Pane 1: connector  # Rust Binance Futures 网关
Pane 2: live bot   # Python live_tick_mm.py
```

## 3. 数据链路

### 3.1 Tardis 历史数据链路

`pipeline.py` 负责：

1. 查找 Tardis `trades` 和 `incremental_book_L2` 文件。
2. 调用 `hftbacktest.data.utils.tardis.convert` 转成标准 `Event` 数组。
3. 可选严格校验 `exch_ts` 和 `local_ts` 递增。
4. 写出单日 `.npz`。
5. 用 `create_last_snapshot` 生成 EOD snapshot。
6. 将 EOD snapshot 串到下一日 SOD。
7. 生成跨日 manifest。

manifest 的关键字段：

```json
{
  "symbol": "BTCUSDT",
  "start_day": "YYYY-MM-DD",
  "end_day": "YYYY-MM-DD",
  "data_files": ["...npz"],
  "initial_snapshot": "...npz or null",
  "latest_eod_snapshot": "...npz"
}
```

### 3.2 实盘原始行情链路

`pipeline_live_raw.py` 解决的是另一件事：用 AWS collector 实际录到的 gzip 作为回测行情，而不是用 Tardis 历史数据。

这样可以消除一个很大的不确定因素：Tardis 数据和 live bot 实际接收的行情流可能在采样、延迟、深度融合、事件缺失上不同。

`pipeline_live_raw.py` 做的事：

- 读取 collector gzip。
- 调用 `hftbacktest.data.utils.binancefutures.convert` 转成标准 `Event`。
- 写出 `.npz`。
- 生成与 `backtest_tick_mm.py` 兼容的 manifest。
- 对 truncated gzip 做容错：遇到缺失 gzip trailer 或不完整尾行时，重写完整换行记录到临时 gzip 再转换。

这个能力已经由 `test_pipeline_live_raw.py` 覆盖，测试点包括：

- 能写出含 `data` key 的 `.npz`。
- manifest 符合 backtest contract。
- 能容忍缺失 gzip trailer。
- 能忽略不完整最终 JSON 行。

## 4. Audit 体系

`audit_schema.py` 定义回测和实盘共享的 audit 字段。字段可以分为几类：

- 标识：`run_id`, `symbol`, `strategy_seq`, `event_type`, `event_source`, `event_seq`
- 时间：`ts_local`, `ts_exch`, `req_ts`, `exch_ts`, `resp_ts`
- 订单动作：`order_id`, `action`, `planned_order_id`, `planned_action`, `throttle_reason`, `reject_reason`
- 延迟：`entry_latency_ns`, `resp_latency_ns`, `spike_flag`, `auditlatency_ms`, `feed_latency_ns`, `latency_signal_ms`
- 市场状态：`best_bid`, `best_ask`, `mid`, `spread_bps`, `vol_bps`
- 策略状态：`fair`, `reservation`, `half_spread`, `position`, `inventory_score`
- 风控/gating：`dropped_by_latency`, `dropped_by_api_limit`, `pos_limit`, `impact_cost`
- Greeks：`greek_delta`, `greek_gamma`, `greek_vega`, `greek_theta`, `greek_adjustment`
- 报价状态：`target_bid_tick`, `target_ask_tick`, `working_bid_tick`, `working_ask_tick`, `working_buy_order_id`, `working_sell_order_id`, `extra_order_ids`, `extra_order_sides`, `extra_order_price_ticks`
- 实盘安全快照：`local_open_orders`, `rest_position`, `position_mismatch`, `rest_open_order_count`, `local_open_order_count`, `rest_open_orders`, `open_order_diff`, `safety_status`, `safety_detail`
- 订单生命周期：`client_order_id`, `exchange_order_id`, `order_side`, `order_price`, `order_price_tick`, `order_qty`, `order_remaining_qty`, `order_executed_qty`, `order_status`, `order_time_in_force`, `lifecycle_state`, `linked_strategy_seq`, `linked_action`, `linked_order_id`, `cancel_requested`, `cancel_request_ts`, `cancel_ack_ts`, `fill_ts`, `fill_qty`, `fill_price`, `fill_trade_id`, `fill_after_cancel_request`, `ws_event_time`, `rest_update_time`, `local_order_seen`, `rest_order_seen`, `ws_order_seen`, `lifecycle_detail`

当前事件类型分为两层：

- `decision`：策略决策行，也是 `compare_audit.py` 计算 action/reject/fair/reservation/position MAE 的对齐行。
- lifecycle/diagnostic rows：`safety_check`, `order_submit_sent`, `cancel_sent`, `order_new`, `order_update`, `cancel_ack`, `fill`, `partial_fill`, `expired`, `rejected`。这些行保留订单生命周期证据，但不会混入决策行对齐指标。

`compare_audit.py` 输出：

- 回测和实盘各自的 latency/drop/inventory/spread/vol 分布。
- 按 `strategy_seq` 对齐后的 action match rate。
- reject reason match rate。
- `fair/reservation/half_spread/position/inventory_score/spread_bps/vol_bps` 的 MAE。
- cadence 统计：
  - 每边 `ts_local` 行数、首尾时间、delta 分布。
  - live timestamp 到最近 backtest timestamp 的 lag 分布。

这里 `strategy_seq` 是核心对齐键。它假设第 N 次策略决策可以在回测和实盘之间比较。如果回测和实盘决策行数差异很大，单靠 `strategy_seq` 比较会混入 cadence 偏差。

## 5. 对齐工作流

实际推荐流程如下。

### 5.1 AWS 东京实盘运行

1. 编译 connector/collector。
2. 配置 Binance Futures API key。
3. 用小仓位 live config 启动：

```bash
cd examples/binance_tick_mm/deploy
./run_live.sh ../config_live.toml ./my_binancefutures.toml BTCUSDT
tmux attach -t hft_live
```

实盘产生两类关键产物：

- `audit_live.csv`: 决策、动作、延迟、盘口、仓位、风控状态。
- collector gzip: live bot 同期实际市场数据。

### 5.2 本地准备延迟模型

```bash
python examples/binance_tick_mm/latency_from_audit.py \
  --audit-csv audit_live.csv \
  --output-npz live_order_latency.npz \
  --output-stats live_order_latency_stats.json
```

脚本从 `req_ts/exch_ts/resp_ts` 抽取有效订单延迟，排序后生成 hftbacktest `IntpOrderLatency` 所需 dtype：

```text
req_ts, exch_ts, resp_ts, _padding
```

它还支持 clip 和 spike simulation，用来把实盘极端长尾映射成可控的回测延迟分布。

### 5.3 本地准备行情

如果目标是普通历史回测，用 Tardis：

```bash
python examples/binance_tick_mm/pipeline.py \
  --tardis-dir ~/data/tardis/binance-futures \
  --out-dir ./out/binance_tick_mm \
  --symbol BTCUSDT \
  --start-day 2026-04-28 \
  --end-day 2026-04-28 \
  --tick-size 0.1 \
  --lot-size 0.001 \
  --snapshot-mode ignore_sod \
  --strict-timestamps
```

如果目标是复现实盘窗口，优先用 collector 原始行情：

```bash
python examples/binance_tick_mm/pipeline_live_raw.py \
  --input-gz /path/to/btcusdt_20260428.gz \
  --out-dir ./out/live_raw \
  --symbol BTCUSDT \
  --start-day 2026-04-28 \
  --end-day 2026-04-28
```

### 5.4 同窗口回测

用 live audit 的首尾 `ts_local` 做绝对切片：

```bash
HFTBACKTEST_USE_LOCAL_PY=1 python examples/binance_tick_mm/backtest_tick_mm.py \
  --config /path/to/config_backtest.toml \
  --manifest /path/to/manifest_2026-04-28_to_2026-04-28.json \
  --window full_day \
  --slice-ts-local-start <live_first_ts_local> \
  --slice-ts-local-end <live_last_ts_local>
```

如果要强制回测按实盘决策时间表运行，使用：

```toml
[backtest_cadence]
mode = "audit_replay"
enabled = true
audit_csv = "/path/to/audit_live.csv"
run_id = "live_btcusdt_..."
ts_column = "ts_local"
tolerance_ms = 0.0
```

`audit_replay` 会加载 live audit decision rows 的 `ts_local` 列，回测只在当前时钟追上下一次 live 决策时间时运行策略决策。旧版没有 `event_type` 列的 CSV 会按兼容模式读取全部 rows。它的作用是把“回测看到了更多 feed event 导致决策次数更多”的问题单独拿出来控制。

`audit_replay` 还支持：

- `replay_mode = "single"`：每个 feed event 最多消费一个 live 决策时间，避免一次性 drain backlog 造成策略状态跳变；iter1 使用这个模式。
- `replay_mode = "drain_due"`：向后兼容的 drain 模式，会在当前 feed event 消费所有已到期 live 决策时间。
- `max_lag_ms`：可选的 lag breach 统计阈值。

回测结果 JSON 会记录 `audit_replay_scheduled_count`, `audit_replay_schedule_stats`, `audit_replay_consumed_count`, `audit_replay_unconsumed_count`, `audit_replay_skipped_due_count`, `audit_replay_max_lag_breaches`, `audit_replay_lag_ns`，用于验收 cadence 是否真的对齐。

### 5.5 对齐报告

```bash
python examples/binance_tick_mm/compare_audit.py \
  --bt /path/to/audit_bt.csv \
  --live /path/to/audit_live.csv \
  --out /path/to/alignment_report.json
```

核心读数：

- `action_match_rate`: 回测和实盘是否做了同样动作。
- `reject_reason_match_rate`: 延迟/API/风控 gating 是否一致。
- `drop_latency_rate`: latency guard 行为是否一致。
- `drop_api_rate`: token bucket/API interval 行为是否一致。
- `fair/reservation MAE`: 市场状态和信号计算是否一致。
- `position MAE`: fill path 和仓位路径是否一致。
- `cadence.nearest_lag_live_to_bt`: 时间节奏是否接近。

## 6. 当前实盘样本：`live_btcusdt_1777342116`

本地分析目录：

```text
local_live_analysis/live_btcusdt_1777342116/
```

这是 2026-04-28 的一次两阶段对齐实盘样本。它的意义是：已经有了 live audit、live 原始行情、延迟模型和同窗口本地回测，可以作为当前最重要的对齐基线。

### 6.1 Live run 概况

来自 `live_summary.md/json`：

- run_id: `live_btcusdt_1777342116`
- rows: `210,110`
- duration: `5,679.4` 秒，约 94.7 分钟
- first `ts_local`: `1777342117432305664`
- last `ts_local`: `1777347796789512192`
- final position/rest: `-0.002 / -0.002`
- final safety status: `open_order_mismatch_pending`

仓位安全结果：

- `position_mismatch max = 0.001`
- `bad_gt_0.003 = 0`
- `critical_position_mismatch = 0`

说明 Binance private `ACCOUNT_UPDATE` 方向的仓位对齐是有效的。

停止或异常点主要在 open-order state：

- local open orders 和 REST open orders 曾出现 drift。
- 该 run 未订阅 `ORDER_TRADE_UPDATE`，因此订单生命周期分析只能诊断，不能作为完整 ground truth。

### 6.2 Live run 风控和延迟

reject/gating：

- latency_guard: `46,716` 行，rate `0.2223`
- quote throttle: `26,386` 行，rate 约 `0.1447`
- api_interval_guard: `4,013` 行，rate `0.0191`

延迟分布：

- feed latency p50 `3.17ms`, p90 `18.39ms`, p99 `363.76ms`
- entry latency p50 `3.00ms`, p90 `587.36ms`, p99 `3697.24ms`
- response latency p50 `2.00ms`, p90 `9.40ms`, p99 `494.71ms`

这里的长尾非常明显。实盘 gating 用 feed latency，回测用 feed latency 与 predicted entry latency 的组合；如果 latency model 的长尾形状和实盘 guard 信号不一致，会直接导致 reject reason 对齐变差。

### 6.3 P2：live raw 转换

P2 已完成：

- 原始 gzip: `raw_market_data/btcusdt_20260428.gz`
- 转换脚本: `examples/binance_tick_mm/pipeline_live_raw.py`
- NPZ: `out/live_raw/btcusdt/btcusdt_20260428.npz`
- manifest: `out/live_raw/btcusdt/manifest_2026-04-28_to_2026-04-28.json`
- NPZ rows: `29,491,708`
- local timestamp coverage:
  - `1777335329858658560`
  - `1777360821081938432`

注意：这个 gzip 有 truncated trailer 问题，但 converter 已经通过“只保留完整换行记录”的方式完成转换。

### 6.4 P3：同窗口回测

P3 使用 live raw manifest，并按 live audit 窗口做绝对 `ts_local` 切片。

切片：

- start: `1777342117432305664`
- end: `1777347796789512192`
- slice rows: `6,952,762`

结果：

- backtest rows: `323,713`
- live rows: `210,110`
- common strategy sequence rows: `210,110`
- action match rate: `0.8490`
- reject reason match rate: `0.4513`
- backtest/live latency drop rate: `0.2613 / 0.2223`
- backtest/live API drop rate: `0.1116 / 0.1447`
- MAE:
  - fair: `107.39`
  - reservation: `107.39`
  - position: `0.00226`
  - spread_bps: `0.0191`
  - vol_bps: `0.1035`

解释：

- 同窗口 market-data replay 已经打通，问题不再是“缺数据”或“时间段错了”。
- 回测行数明显多于实盘，说明回测在更多 feed event 上做了决策。
- action match 已经达到约 85%，证明策略核心复用有效。
- reject reason 和 fair/reservation 差异仍大，主要剩余问题在 cadence、latency guard/API timing 和 fill/order lifecycle。

### 6.5 固定 cadence sweep

固定最小决策间隔测试了 `0, 2, 5, 8, 10, 15, 20ms`。

关键结论：

- `0ms`: rows `323,713`, action match `0.8490`
- `2ms`: rows `165,625`, action match `0.7918`
- `20ms`: rows `116,988`, action match `0.7430`

固定 interval 并没有解决 cadence 对齐：

- interval 增大后 rows 反而低于 live。
- fair/spread MAE 有改善，但 action path 变差。
- 说明 live 决策节奏不是一个简单 hard minimum interval，而是具有 burst/gap 结构。

### 6.6 Plan B：audit-derived cadence replay

`audit_replay` 结果：

- scheduled live decisions: `210,110`
- consumed decisions: `210,109`
- unconsumed: `1`
- backtest/live rows: `210,109 / 210,110`
- common strategy sequence rows: `210,109`
- action match rate: `0.8442`
- reject reason match rate: `0.5119`
- backtest/live latency drop rate: `0.2114 / 0.2223`
- backtest/live API drop rate: `0.1183 / 0.1447`
- nearest live-to-backtest cadence lag p99: `24.7ms`
- MAE:
  - fair: `12.11`
  - reservation: `12.11`
  - position: `0.00181`
  - spread_bps: `0.0196`
  - vol_bps: `0.0906`

解释：

- 行数基本对齐，fair/reservation MAE 大幅下降。
- reject reason match 从 `0.4513` 提升到 `0.5119`。
- action match 从 `0.8490` 小幅降到 `0.8442`，说明仅修正 cadence 不能解决动作路径。
- 下一瓶颈更可能是 latency/API timing、订单生命周期和 fill path，而不是单纯决策次数。

## 7. 当前主要缺口

### 7.1 Open-order lifecycle ground truth 不完整

`TODO_live_open_order_alignment.md` 和 Plan C 文档指出，`live_btcusdt_1777342116` 的停止原因是 open-order state drift，不是 position mismatch。

确认点：

- `ACCOUNT_UPDATE` 修复后仓位对齐良好。
- 当前问题是 local open orders 与 REST open orders 分歧。
- 当时 connector 只订阅了 `ACCOUNT_UPDATE`，未订阅 `ORDER_TRADE_UPDATE`。

计划方向：

- Binance futures private stream 同时订阅：
  - `ACCOUNT_UPDATE`
  - `ORDER_TRADE_UPDATE`
- `ORDER_TRADE_UPDATE` 进入 `OrderManager::update_from_ws`，提供订单生命周期 WebSocket ground truth。
- 不应放宽 open-order safety；应先补齐订单事件。

这个缺口会直接影响：

- 实盘本地订单表。
- `working_bid_tick/working_ask_tick`。
- cancel/submit 决策。
- 后续与回测的 action path 比较。

### 7.2 Latency model 与 gating 信号仍需拆分

当前 live 和 backtest 对 latency signal 的定义不同：

- live: `latency_signal_ns = feed_latency_ns`
- backtest: `latency_signal_ns = max(feed_latency_ns, predicted_entry_ns)`

这个差异是有意的：live 的历史 order latency 不是前瞻可知信号，且 REST/WebSocket order timestamp 可能滞后，不适合直接拿来 gate。但回测为了模拟实盘订单延迟，需要 predicted entry latency。

后续需要分清两类问题：

- 用于模拟订单到达/响应的 `IntpOrderLatency`。
- 用于策略当下是否跳过报价的 latency guard signal。

如果二者混在一起，会导致“成交路径更真实”和“决策 gating 更像 live”互相拉扯。

### 7.3 Fill path 和 queue 参数还没有闭环校准

当前回测使用 `NoPartialFillExchange + PowerProbQueueModel3(5)`。这能快速跑通，但和实盘成交路径仍可能不同：

- 实盘订单可能部分成交、取消竞态、REST ACK 与 WS update 顺序不同。
- 回测 `NoPartialFillExchange` 不模拟部分成交。
- `power_prob_n` 影响同价队列推进和 fill rate，需要用实盘 fill/position 路径校准。

在 `ORDER_TRADE_UPDATE` 接入前，fill path 校准的 ground truth 不完整。

### 7.4 配置和实现处于演进状态

`config.example.toml` 是通用模板；`local_live_analysis/live_btcusdt_1777342116/config_*_two_phase_align.toml` 记录了更贴近该次 live run 的两阶段对齐配置，例如：

- quote throttle
- two-phase replace
- live safety
- REST position safety
- max position qty

这些字段在分析产物中已经出现，但需要持续确认当前 `examples/binance_tick_mm` 入口脚本是否完全消费这些配置。做下一轮实验前，建议先固定一个“当前真实使用的 live/backtest config contract”，避免文档配置、实盘配置和脚本读取逻辑漂移。

## 8. 策略优化前的建议门槛

建议把工作分成两层：先做复现，再做优化。

### 8.1 复现层目标

优先目标：

- 同窗口使用 live raw manifest，而不是不同源/不同日数据。
- 回测和实盘 rows 接近，或明确用 `audit_replay` 控制决策时刻。
- `action_match_rate` 稳定超过 85% 后继续提升。
- `reject_reason_match_rate` 从当前约 51% 继续提高。
- latency/API drop rate 差异收敛。
- position MAE 下降，并且 final position path 无系统偏差。
- open-order drift 由 `ORDER_TRADE_UPDATE` 修复后不再触发安全停止。

可以接受的阶段性判断：

- 如果 `fair/reservation MAE` 大但 action match 高，优先看 market/cadence。
- 如果 row count 对齐后 action match 仍停在 84% 左右，优先看订单状态和 latency/API gating。
- 如果 position MAE 大或方向偏，优先看 queue/fill model。

### 8.2 优化层工具

复现达到基本可用后，再使用：

- `walk_forward.py`: train/test 滚动评估，避免只对单日过拟合。
- `sweep_backtest.py`: 参数网格并行扫描，建议配合 `audit.mode = "off"` 和 summary 输出。
- `backtest_metrics.py`: 用 summary JSON/daily CSV 降低 I/O。
- `hash_audit.py`: 同数据同参数确定性检查。
- `validate_audit.py`: 确保 audit schema 和公式没有破坏。

大规模扫描时建议重点参数：

- `[risk] base_spread`
- `[risk] k_inv`
- `[risk] k_pos`
- `[fair] w_imb`
- `[queue] power_prob_n`
- `[latency] latency_guard_ms`
- `[api_limit] min_interval_ms`

但不要在 open-order lifecycle 未修复前过度优化 fill rate，否则可能把实盘订单状态缺口拟合进参数里。

## 9. 推荐下一步

按工程依赖顺序，下一步建议是：

1. 完成 Plan C：connector 同时订阅 `ORDER_TRADE_UPDATE`。
2. 跑短 live validation，确认 submit/cancel/fill 都有订单 WebSocket update。
3. 再跑一段 1-2 小时小仓位 live，确认：
   - position mismatch 仍安全。
   - open-order mismatch 不再持续。
   - audit_live 和 raw market data 完整落盘。
4. 用 `pipeline_live_raw.py` 转换该 run 的 collector gzip。
5. 用 live 首尾 `ts_local` 做同窗口回测。
6. 分别跑：
   - normal cadence
   - `audit_replay`
7. 用 `compare_audit.py` 比较：
   - action match
   - reject reason match
   - latency/API drop
   - position path
   - cadence lag
8. 如果订单生命周期已稳定，再校准：
   - latency guard signal
   - `IntpOrderLatency` 分布
   - queue/fill 参数
9. 最后进入 walk-forward 和参数 sweep。

一句话总结：`examples/binance_tick_mm` 已经从策略示例演进成一套“实盘可观测、行情可回放、延迟可注入、决策可对齐、参数可扫描”的实验系统。当前最有价值的成果是 live raw 同窗口回放和 audit-derived cadence replay 已经打通；当前最关键的阻塞是订单生命周期 ground truth 仍需通过 `ORDER_TRADE_UPDATE` 补齐，然后再做延迟和队列模型校准。
