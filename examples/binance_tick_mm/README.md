# Binance Tick MM Backtest (ROIVector)

本目录实现了你指定的合并方案：

- 引擎：`ROIVectorMarketDepthBacktest + NoPartialFillExchange + PowerProbQueueModel3(5)`
- 策略：Greeks fair price + 简化 AS 库存管理 + 线性冲击成本
- 风控：延迟尖刺仿真、`entry>5ms` 拒单、API 令牌桶限流、`API<20ms` 跳过、超仓只挂减仓侧
- 审计：`audit_bt` 与 `audit_live` 同 schema（见 `audit_schema.py`）
- 数据流：Tardis `incremental_book_L2 + trades` -> snapshot -> npz -> 严格时序校验 -> 跨日 manifest

## Files

- `config.example.toml`：配置模板
- `pipeline.py`：Tardis 数据转换 + snapshot 链 + strict timestamp 校验
- `latency_from_audit.py`：由实盘 `audit_live.csv` 生成 `IntpOrderLatency npz`
- `backtest_tick_mm.py`：主回测与审计输出
- `run_env_test.py`：Mac/amdserver 执行入口
- `walk_forward.py`：滚动 walk-forward 回测（训练N天+测试M天）
- `audit_schema.py`：统一审计字段定义
- `compare_audit.py`：`audit_bt` vs `audit_live` 对齐报告
- `validate_audit.py`：审计字段与关键口径校验
- `hash_audit.py`：回归哈希（确定性检查）
- `plot_audit.py`：从 audit 输出 `returns/position` 图片
- `sync_to_amdserver.sh`：同步脚本（到 `amdserver:~/project/hftbacktest`）

## Prerequisite

建议使用独立 conda 环境：

```bash
conda create -y -n hftbacktest python=3.12 pip
conda activate hftbacktest
pip install hftbacktest==2.4.4
```

amdserver 可用：

```bash
/home/molly/anaconda3/bin/conda create -y -n hftbacktest python=3.12 pip
source /home/molly/anaconda3/etc/profile.d/conda.sh
conda activate hftbacktest
pip install hftbacktest==2.4.4
```

## 1) 从实盘审计生成延迟模型

```bash
cd /Users/liu/Documents/hftbacktest/examples/binance_tick_mm
python latency_from_audit.py \
  --audit-csv /path/to/audit_live.csv \
  --output-npz /Users/liu/Documents/hftbacktest/out/binance_tick_mm/live_order_latency.npz \
  --output-stats /Users/liu/Documents/hftbacktest/out/binance_tick_mm/live_order_latency_stats.json \
  --entry-ms-min 1.2 --entry-ms-max 2.8 \
  --resp-ms-min 1.0 --resp-ms-max 2.2 \
  --spike-prob 0.01 --spike-ms-min 8.0 --spike-ms-max 10.0
```

把生成的 npz 填到 `config.toml` 的 `latency.order_latency_npz`。

Greeks 参数在 `config.toml` 的 `[greeks]`：

- `enabled`：是否启用显式 Greeks 校正
- `signal_csv`：可选时序文件（列：`ts_local,delta,gamma,vega,theta`）
- `use_position_as_delta=true` 时，未提供 `signal_csv` 会用当前仓位作为 `delta`
- `w_delta/w_gamma/w_vega/w_theta`：fair price Greeks 权重
- `scale_delta/scale_gamma/scale_vega/scale_theta`：信号标准化/缩放系数

## 2) Mac 功能测试（BTC 单日前 5 分钟）

数据目录按你的要求支持：

- 首选：`~/document/tardis`
- 自动兼容：`~/Documents/tardis`

```bash
cd /Users/liu/Documents/hftbacktest/examples/binance_tick_mm
cp config.example.toml config.toml
# 编辑 config.toml：start_day/end_day、latency.order_latency_npz、output_root 等

python run_env_test.py \
  --config ./config.toml \
  --target mac \
  --day 2025-01-01 \
  --plot
```

输出：

- manifest：`<output_root>/mac/btcusdt/manifest_*.json`
- 审计：`<output_root>/mac/audit_bt_mac_YYYY-MM-DD.csv`
- 图片：`<output_root>/mac/plots/*_returns.png`、`*_position.png`

## 3) amdserver 长时测试（全日或数小时）

先同步代码：

```bash
cd /Users/liu/Documents/hftbacktest/examples/binance_tick_mm
./sync_to_amdserver.sh
```

服务器执行（数据目录：`~/data/tardis/binance-futures`）：

```bash
cd ~/project/hftbacktest/examples/binance_tick_mm
python run_env_test.py --config ./config.toml --target amdserver --day 2025-01-01
```

若太慢，先跑快速窗口：

```bash
python run_env_test.py --config ./config.toml --target amdserver --day 2025-01-01 --fast
```

## 4) 7 天跨日数据链路

```bash
cd /Users/liu/Documents/hftbacktest/examples/binance_tick_mm
python pipeline.py \
  --tardis-dir ~/Documents/tardis \
  --out-dir /Users/liu/Documents/hftbacktest/out/binance_tick_mm \
  --symbol BTCUSDT \
  --start-day 2025-01-01 \
  --end-day 2025-01-07 \
  --tick-size 0.1 \
  --lot-size 0.001 \
  --snapshot-mode ignore_sod \
  --strict-timestamps
```

该流程会：

- 转换 `trades + incremental_book_L2` 到标准 `npz`
- 校验 `exch_ts/local_ts` 严格递增，失败即 fail-fast
- 失败时输出 `strict_report_*.json`（包含 non-strict pair 统计）
- 日末产出 `EOD snapshot`，自动串到下一日 `SOD`
- 生成跨日 `manifest_YYYY-MM-DD_to_YYYY-MM-DD.json`

补充：若本地是通道目录单文件结构（例如 `trades/BTCUSDT.csv.zst`、`incremental_book_L2/BTCUSDT.csv.zst`），
单日测试会自动兼容；多日回放建议使用带日期后缀的分日文件。

## 5) 回测/实盘对齐报告

```bash
cd /Users/liu/Documents/hftbacktest/examples/binance_tick_mm
python compare_audit.py \
  --bt /path/to/audit_bt.csv \
  --live /path/to/audit_live.csv \
  --out /path/to/audit_alignment_report.json
```

报告包含：

- 延迟分布
- 延迟/限流弃单率
- 库存健康度统计
- 基于 `strategy_seq` 的 action/reject 一致率
- `fair/reservation/half_spread/position/spread_bps/vol_bps` 的 MAE

## 6) Walk-Forward（几十天更可信评估）

```bash
cd /Users/liu/Documents/hftbacktest/examples/binance_tick_mm
python walk_forward.py \
  --config ./config.toml \
  --target mac \
  --start-day 2025-01-01 \
  --end-day 2025-03-31 \
  --train-days 7 \
  --test-days 1 \
  --window full_day \
  --plot
```

输出：

- `.../walk_forward/<target>/walk_forward_summary.csv`
- `.../walk_forward/<target>/walk_forward_summary.json`
- 每个 fold 的 train/test audit 与（可选）test 图

## 7) 回归确定性（同数据同参数）

```bash
cd /Users/liu/Documents/hftbacktest/examples/binance_tick_mm
python hash_audit.py --file /path/to/audit_bt.csv
```

将输出 SHA256，可用于 CI 或本地回归比对。

## 8) 审计字段自检

```bash
cd /Users/liu/Documents/hftbacktest/examples/binance_tick_mm
python validate_audit.py --file /path/to/audit_bt.csv --strict
```

## 9) 大规模回测加速（摘要模式 + 参数并行扫描）

当你要做 30 天以上、上百组参数的批量实验时，建议关闭逐事件审计，改为仅输出汇总指标：

- `[audit] mode = "off"`（可选：`full | actions_only | sampled | off`）
- `[audit] sample_every = 100`（仅在 `mode = "sampled"` 时生效）
- `[summary] enabled = true`
- `[summary] output_json = true`
- `[summary] daily_csv = true`

基线实测（同一台机器）：10 天、`audit=full` 约 `1532.3 sec`，审计 CSV 约 `22GB`。
大规模模式下只写 summary JSON + daily CSV，显著降低 I/O 与磁盘占用，更适合参数扫描。

批量参数并行扫描命令：

```bash
cd /home/molly/project/hftbacktest/examples/binance_tick_mm
python sweep_backtest.py \
  --base-config ./config.toml \
  --manifest /path/to/manifest_2025-01-01_to_2025-01-30.json \
  --grid ./sweep.toml \
  --workers 8 \
  --window full_day \
  --out /home/molly/project/hftbacktest/out/binance_tick_mm/sweeps
```

`sweep.toml` 示例（每个数组取笛卡尔积，自动生成参数组合）：

```toml
[risk]
k_inv = [1.0, 1.5, 2.0]
base_spread = [8.0, 10.0, 12.0]
k_pos = [0.5, 0.75, 1.0]

[fair]
w_imb = [0.2, 0.4, 0.6]
```

输出建议至少包含：

- 每组参数 1 份汇总 JSON（PnL、Sharpe、最大回撤、成交统计）
- 按日 CSV（便于按交易日过滤异常段、做稳定性筛选）

## 实盘部署

### 前提条件

- Rust 工具链（`rustup`、`cargo`）
- Binance Futures API key，需开启交易权限
- Linux 服务器（推荐亚洲节点，降低延迟）
- `tmux` 已安装
- Python 环境已安装 `hftbacktest`（同上 Prerequisite）

### 编译 connector 和 collector

```bash
cd /home/molly/project/hftbacktest/connector && cargo build --release
cd /home/molly/project/hftbacktest/collector && cargo build --release
```

编译产物位于 `connector/target/release/connector` 和 `collector/target/release/collector`。

### 配置

1. 复制并填写 connector 配置：

```bash
cp deploy/binancefutures.toml deploy/my_binancefutures.toml
# 编辑 my_binancefutures.toml，填入 api_key 和 secret
```

2. 复制并调整策略配置：

```bash
cp config.example.toml config_live.toml
```

重点修改项：

| 配置项 | 说明 | 建议值 |
|--------|------|--------|
| `[risk] order_notional` | 单笔下单名义值 | `100`（小仓位测试） |
| `[risk] max_notional_pos` | 最大持仓名义值 | `1000`（严格限制） |
| `[live] connector_name` | connector 启动时的 `--name` 参数 | `"bf"` |
| `[live] roi_lb / roi_ub` | 当前 BTC 价格范围 | 根据实际行情设定 |
| `[paths] output_root` | 审计输出目录 | 服务器上的绝对路径 |

### 启动

```bash
cd /home/molly/project/hftbacktest/examples/binance_tick_mm/deploy
./run_live.sh ../config_live.toml ./my_binancefutures.toml BTCUSDT
tmux attach -t hft_live
```

`run_live.sh` 会在 tmux session `hft_live` 中启动三个 pane：

- Pane 0：collector（行情采集）
- Pane 1：connector（交易所网关）
- Pane 2：live bot（策略主进程）

### 数据回流与校准

实盘运行后，将 `audit_live.csv` 回传到回测服务器，进行回测对齐：

```bash
# 1. 从实盘服务器拷贝审计文件
scp tokyo:/path/to/audit_live.csv ./

# 2. 生成延迟模型
python latency_from_audit.py \
  --audit-csv audit_live.csv \
  --output-npz ./out/binance_tick_mm/live_order_latency.npz

# 3. 对同一时间段跑回测
python pipeline.py --config config.toml
python backtest_tick_mm.py --config config.toml --manifest manifest.json

# 4. 对齐比较
python compare_audit.py \
  --bt ./out/binance_tick_mm/audit_bt.csv \
  --live audit_live.csv
```

### 校准优先级

首次上线后按以下顺序调优：

| 优先级 | 模块 | 调优内容 | 说明 |
|--------|------|----------|------|
| 1 | 延迟 | `latency_from_audit.py` 参数 | 用实盘 audit 拟合延迟分布，是对齐的前提 |
| 2 | 风控 | `order_notional`、`max_notional_pos` | 确认风控阈值与实际滑点匹配 |
| 3 | 费率 | `[fee] maker/taker` | 确认实际费率等级 |
| 4 | 队列 | `power_prob_n` | 对齐成交率（fill rate） |
| 5 | 公允价 | `[fair]` 权重、`[greeks]` 参数 | 微调 fair price 偏差 |
