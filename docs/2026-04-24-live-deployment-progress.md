# 2026-04-24 实盘部署与回测对齐进展

## 目标

让高频做市策略的本地回测表现与实盘表现一致（backtest-live alignment），从而可以在本地回测环境中可信地迭代策略。

## 今日完成的工作

### 1. 代码重构：提取共享策略模块

- 从 `backtest_tick_mm.py` 提取共享逻辑到 `strategy_core.py`（EwmaSigma、TokenBucket、GreekOracle、下单决策、审计行构建等）
- 回测和实盘代码复用同一策略核心，确保逻辑一致
- 相关 commit: `6d7443d`, `2665bab`

### 2. 编写实盘做市引擎

- 新建 `live_tick_mm.py`，使用 `ROIVectorMarketDepthLiveBot` + `LiveInstrument` 连接 Rust connector
- 策略逻辑与回测完全一致，仅替换数据接口层
- 支持 SIGINT/SIGTERM 优雅退出（自动撤销所有挂单）
- 每 60 秒输出 heartbeat（仓位、中间价、spread、vol、feed 延迟）
- 写出与回测相同 schema 的 `audit_live.csv`（44 字段）
- 相关 commit: `b2f78d5`

### 3. 部署配置

- 新增 `config.example.toml` 的 `[live]` 段（connector_name、roi_lb/ub、audit 输出等）
- 新增 `deploy/binancefutures.toml`（Binance Futures 生产环境 connector 配置模板）
- 新增 `deploy/run_live.sh`（tmux 一键启动 collector + connector + bot）
- 相关 commit: `985e0ec`, `110fda1`

### 4. 东京服务器（awsserver1）配置

- 安装 Rust 1.95.0 toolchain
- 编译 connector 和 collector（`cargo build --release`）
- 创建 Python venv，从源码编译安装 hftbacktest 2.4.4（启用 `live` feature）
- 修复上游 bug：`binding.py` 中 `hashmaplive_modify`/`roiveclive_modify` 符号缺失导致 live feature 无法加载（commit: `bb4864d`）
- 配置 API keys、创建 `config_live.toml`（小仓位：order_notional=100, max_notional=1000）

### 5. 实盘启动与验证

- 三组件（collector + connector + live bot）在 tmux session `hft_live` 中运行
- collector 正常录制行情数据（`~/hft_live/data/btcusdt_YYYYMMDD.gz`）
- connector 正常连接 Binance WebSocket + REST API
- live bot 正常接收行情、执行策略、写出 audit_live.csv

### 6. Bug 修复：latency_guard 误杀

- **问题**：`latency_signal_ns` 包含了 `last_resp_ns`（resp_ts - exch_ts），在实盘中这个值包含 REST→WebSocket 传播延迟（~1-2 秒），导致 99.97% 的订单被 `latency_guard`（5ms）误拦
- **修复**：从 `latency_signal_ns` 计算中移除 `last_resp_ns`，仅使用 `feed_latency_ns` 和 `last_entry_ns`
- **效果**：修复后策略正常交易，仓位开始变化
- 相关 commit: `babd942`

### 7. 校准流程验证（方案 B）

在本地用已有 Tardis 历史数据（2026-04-10）验证了完整校准管道：

```
audit_live.csv → latency_from_audit.py → live_order_latency.npz
                                              ↓
Tardis 数据 → pipeline/convert → NPZ → backtest_tick_mm.py → audit_bt.csv
                                              ↓
                                    compare_audit.py → alignment_report.json
```

校准结果（不同日数据，仅验证流程通畅）：
- action_match_rate: **99.95%**（策略逻辑高度一致）
- reject_reason_match_rate: **99.84%**
- half_spread MAE: **0.00002**（几乎完全一致）
- fair MAE: 6499（预期差异，不同日 BTC 价格差 ~$2K）

实测延迟分布：
- entry latency 均值: **2.87ms**
- resp latency 均值: **2.27ms**
- spike 比例: **1.06%**

## 当前状态

实盘 bot 正在 awsserver1 上运行：

| 项目 | 值 |
|------|-----|
| tmux session | `hft_live` |
| run_id | `live_btcusdt_1776997002` |
| 交易对 | BTCUSDT |
| 当前仓位 | -0.001 BTC (~$78) |
| BTC 价格 | ~78,235 |
| Feed 延迟 | 1.9-3.2ms |
| 审计行数 | 25,000+（持续增长中） |

### 服务器目录结构

```
awsserver1:~/hft_live/
├── hftbacktest/              # Git repo (master, latest)
│   ├── target/release/
│   │   ├── connector
│   │   └── collector
│   └── examples/binance_tick_mm/
│       ├── live_tick_mm.py   # 运行中
│       ├── audit_live.csv    # 实时写入
│       └── strategy_core.py
├── config/
│   ├── binancefutures.toml   # API credentials
│   └── config_live.toml      # 小仓位配置
├── output/
├── data/                     # collector 录制的行情
│   └── btcusdt_YYYYMMDD.gz
└── venv/                     # Python 虚拟环境
```

## 待办：方案 A 同日对齐

### 步骤

1. **让 bot 继续运行到 2026-04-25**，积累完整一天的 audit_live.csv

2. **下载 Tardis 2026-04-24 数据**（T+1 可用）：
   - `trades/2026/04/24/BTCUSDT.csv.zst`
   - `incremental_book_L2/2026/04/24/BTCUSDT.csv.zst`

3. **停止 bot，拷回数据**：
   ```bash
   # 在 awsserver1 上
   tmux send-keys -t hft_live:main.2 C-c

   # 在本地
   scp awsserver1:~/hft_live/hftbacktest/examples/binance_tick_mm/audit_live.csv ./
   ```

4. **转换 Tardis 数据**（.zst → .gz → NPZ）：
   ```bash
   mkdir -p /tmp/tardis_20260424
   zstd -d ~/data/tardis/binance-futures/trades/2026/04/24/BTCUSDT.csv.zst -o /tmp/tardis_20260424/trades.csv
   gzip /tmp/tardis_20260424/trades.csv
   zstd -d ~/data/tardis/binance-futures/incremental_book_L2/2026/04/24/BTCUSDT.csv.zst -o /tmp/tardis_20260424/depth.csv
   gzip /tmp/tardis_20260424/depth.csv

   python -c "
   from hftbacktest.data.utils.tardis import convert
   convert(['/tmp/tardis_20260424/trades.csv.gz', '/tmp/tardis_20260424/depth.csv.gz'],
           output_filename='/tmp/tardis_20260424/BTCUSDT_20260424.npz',
           buffer_size=500_000_000, ss_buffer_size=5_000_000)
   "
   ```

5. **生成延迟模型 + 跑回测 + 对比**：
   ```bash
   cd examples/binance_tick_mm

   python latency_from_audit.py --audit-csv audit_live.csv \
     --output-npz ./out/calibration/live_order_latency.npz \
     --output-stats ./out/calibration/latency_stats.json

   # 更新 config 中的日期和 latency npz 路径，然后
   python backtest_tick_mm.py --config config.toml --manifest manifest.json

   python compare_audit.py --bt audit_bt.csv --live audit_live.csv \
     --out alignment_report.json
   ```

6. **分析 alignment_report.json**，按优先级调参：

   | 优先级 | 看什么 | 调什么 |
   |--------|--------|--------|
   | 1 | entry_latency 分布 | `[latency]` min/max/spike |
   | 2 | action_match_rate | 整体一致性 |
   | 3 | fair MAE | `[fair]` w_imb, w_spread, w_vol |
   | 4 | position MAE | `[risk]` k_inv, base_spread |
   | 5 | drop_rate 差异 | latency_guard_ms, api_limit |
   | 6 | 成交数量差异 | `[queue]` power_prob_n |

### 对齐目标

- action_match_rate > 90%
- fair MAE < 0.5 tick
- position MAE < 10% of max_notional_pos
- latency 分布 p50/p90/p99 误差 < 10%
- drop rate 差异 < 2%

## Git Commits 总结

| SHA | 描述 |
|-----|------|
| `6d7443d` | refactor: extract strategy_core.py |
| `2665bab` | fix: remove unused import, type expand_path |
| `985e0ec` | feat: add [live] config section |
| `b2f78d5` | feat: add live_tick_mm.py |
| `110fda1` | feat: add deploy config + tmux script |
| `f7df8a8` | docs: add live trading guide to README |
| `bb4864d` | fix: handle missing modify symbols in binding.py |
| `babd942` | fix: exclude resp_latency from latency_signal |
