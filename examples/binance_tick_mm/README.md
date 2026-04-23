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

## 6) 回归确定性（同数据同参数）

```bash
cd /Users/liu/Documents/hftbacktest/examples/binance_tick_mm
python hash_audit.py --file /path/to/audit_bt.csv
```

将输出 SHA256，可用于 CI 或本地回归比对。

## 7) 审计字段自检

```bash
cd /Users/liu/Documents/hftbacktest/examples/binance_tick_mm
python validate_audit.py --file /path/to/audit_bt.csv --strict
```
