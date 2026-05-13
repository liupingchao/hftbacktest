# Binance Tick MM 实盘回测对齐实施计划

本文是 `examples/binance_tick_mm` 下一阶段实盘回测对齐的可执行计划。前提假设：

- Binance Futures connector 的 `ORDER_TRADE_UPDATE` 已在 AWS 东京服务器短测验证通过。
- 后续如果再发现 open-order/position/connector 新 bug，再单独回到故障分析分支。
- AWS 东京服务器登录方式为 `ssh admin@awsserver1`。
- AWS 服务器工作根目录固定为 `/home/admin/hft_live`。
- AWS 上项目路径为 `/home/admin/hft_live/hftbacktest`。
- 实盘采集得到的 gzip 行情文件必须拷贝回本地 Ubuntu 机器，在本地 `local_live_analysis/<RUN_ID>/` 下完成转换、回测、对齐分析和最终归档。

## 0. 总目标

拿到一份订单生命周期完整的新 live baseline，并在本地用同一段实盘行情、同一段时间窗口、同一策略逻辑进行回测复现。之后按 cadence、latency/API、fill/queue、策略参数的顺序逐项校准，最终进入 walk-forward 和参数扫描。

最终希望达到：

- 实盘订单生命周期可由 `ORDER_TRADE_UPDATE` 可靠还原。
- 本地回测可复现实盘同窗口行为。
- 对齐报告指标稳定优于旧基线 `live_btcusdt_1777342116`。
- 策略参数优化建立在可信执行模型上，而不是拟合 connector 或订单状态缺口。

## 1. 阶段 A：确认 AWS 环境和版本

### 操作

登录服务器：

```bash
ssh admin@awsserver1
```

确认项目目录：

```bash
ls -la /home/admin/hft_live
test -d /home/admin/hft_live/hftbacktest
```

进入项目目录：

```bash
cd /home/admin/hft_live/hftbacktest
```

确认 connector 代码包含 `ORDER_TRADE_UPDATE`：

```bash
rg "ORDER_TRADE_UPDATE|UserDataStreamEvent|connect_events_only|Received Binance futures ORDER_TRADE_UPDATE" connector/src/binancefutures
```

重新编译 connector 和 collector：

```bash
cargo build --release -p connector
cargo build --release -p collector
```

确认 binary 时间：

```bash
ls -lh target/release/connector target/release/collector
```

### 验收

- `rg` 能看到：
  - `UserDataStreamEvent::OrderTradeUpdate`
  - `events=ORDER_TRADE_UPDATE`
  - `connect_events_only`
  - `Received Binance futures ORDER_TRADE_UPDATE.`
- `cargo build --release -p connector` 成功。
- `target/release/connector` 更新时间晚于本次部署时间。

### 失败处理

- 如果 AWS 代码不是最新：先同步本地代码到 AWS，再重新 build。
- 如果 build 失败：先停止进入实盘阶段，修复编译问题。

## 2. 阶段 B：启动新的小仓位 live baseline

### 目标

跑一份新的 1-2 小时 live baseline。此阶段目标不是盈利，而是采集干净、可复现的数据。

### 建议配置

小仓位、安全优先：

```toml
[risk]
order_notional = 100.0
max_notional_pos = 250.0
max_position_qty = 0.003

[latency]
latency_guard_ms = 5.0

[api_limit]
enabled = true
capacity = 20.0
refill_per_sec = 20.0
min_interval_ms = 20.0

[live_safety]
enabled = true
open_order_check = true
fail_on_mismatch = true
```

### 操作

在 AWS 上准备 run 目录：

```bash
RUN_ID="live_btcusdt_$(date +%s)"
mkdir -p /home/admin/hft_live/runs/$RUN_ID
mkdir -p /home/admin/hft_live/runs/$RUN_ID/logs
mkdir -p /home/admin/hft_live/runs/$RUN_ID/data
mkdir -p /home/admin/hft_live/runs/$RUN_ID/output
echo "$RUN_ID"
```

复制配置快照：

```bash
cp examples/binance_tick_mm/config_live.toml /home/admin/hft_live/runs/$RUN_ID/config_live.toml
cp /home/admin/hft_live/config/binancefutures.toml /home/admin/hft_live/runs/$RUN_ID/binancefutures.toml
```

启动 tmux：

```bash
cd examples/binance_tick_mm/deploy
DATA_DIR=/home/admin/hft_live/runs/$RUN_ID/data \
./run_live.sh /home/admin/hft_live/runs/$RUN_ID/config_live.toml /home/admin/hft_live/runs/$RUN_ID/binancefutures.toml BTCUSDT
tmux attach -t hft_live
```

运行中观察 connector 日志：

```bash
tmux capture-pane -pt hft_live:main.1 -S -200 | rg "ORDER_TRADE_UPDATE|error|Error|open_order|mismatch"
```

运行 1-2 小时后优雅停止 live bot：

```bash
tmux send-keys -t hft_live:main.2 C-c
```

等待 bot 完成撤单和关闭，再停止 connector/collector：

```bash
tmux send-keys -t hft_live:main.1 C-c
tmux send-keys -t hft_live:main.0 C-c
```

保存 tmux 日志：

```bash
tmux capture-pane -pt hft_live:main.0 -S -50000 > /home/admin/hft_live/runs/$RUN_ID/logs/collector.log
tmux capture-pane -pt hft_live:main.1 -S -50000 > /home/admin/hft_live/runs/$RUN_ID/logs/connector.log
tmux capture-pane -pt hft_live:main.2 -S -50000 > /home/admin/hft_live/runs/$RUN_ID/logs/live_bot.log
```

### 产物

必须保留：

- `/home/admin/hft_live/runs/$RUN_ID/config_live.toml`
- `/home/admin/hft_live/runs/$RUN_ID/binancefutures.toml`
- `/home/admin/hft_live/runs/$RUN_ID/logs/connector.log`
- `/home/admin/hft_live/runs/$RUN_ID/logs/live_bot.log`
- `/home/admin/hft_live/runs/$RUN_ID/data/*.gz`
- `audit_live*.csv`

### 验收

必须满足：

- connector 日志中存在多条：

```text
Received Binance futures ORDER_TRADE_UPDATE.
```

- live bot 正常写出 audit。
- collector gzip 存在且非空。
- 结束时没有持续 open-order mismatch。
- position mismatch 没有触发 critical safety。
- bot 优雅退出时能撤销工作单。

建议门槛：

- live rows > `50,000`，或运行时长 > `60min`。
- final local position 与 REST position 一致，误差 <= `0.001 BTC`。
- open-order mismatch 不在最终状态持续存在。

### 失败处理

- 如果没有 `ORDER_TRADE_UPDATE` 日志：停止对齐，回 connector subscription 检查。
- 如果 open-order mismatch 仍持续：保留 run 目录，分析 REST open orders、local orders、ORDER_TRADE_UPDATE 时序。
- 如果 collector gzip 损坏：可尝试 `pipeline_live_raw.py` 的 truncated gzip 容错；若转换失败，重跑 baseline。

## 3. 阶段 C：回传数据到本地 Ubuntu

### 操作

以下操作在本地 Ubuntu 机器执行。实盘采集的 gzip 行情文件必须回传到本地分析目录，后续转换、回测、对齐报告和归档都在本地完成。

推荐使用已实现的本地编排脚本：

```bash
cd /home/molly/project/hftbacktest
python examples/binance_tick_mm/align_live_run.py --run-id <RUN_ID>
```

该脚本默认会从：

```text
admin@awsserver1:/home/admin/hft_live/runs/<RUN_ID>
```

拉取配置、日志、gzip 和 audit，随后在本地自动执行：

- `latency_from_audit.py` 等价逻辑
- `pipeline_live_raw.py` 等价逻辑
- normal cadence 回测
- `audit_replay` 回测
- `compare_audit.py` 等价逻辑
- `local_live_analysis/archive/<RUN_ID>.tar.gz` 归档

常用选项：

```bash
# 只拉取和准备，不跑耗时回测
python examples/binance_tick_mm/align_live_run.py --run-id <RUN_ID> --skip-backtest

# 已经手动拉取过，只做本地转换/回测/归档
python examples/binance_tick_mm/align_live_run.py --run-id <RUN_ID> --skip-fetch

# 不归档，便于调试
python examples/binance_tick_mm/align_live_run.py --run-id <RUN_ID> --skip-archive

# 只打印 scp 命令，不执行远程拉取
python examples/binance_tick_mm/align_live_run.py --run-id <RUN_ID> --dry-run
```

下面保留手工步骤，便于脚本失败时逐步排查。

本地创建分析目录：

```bash
cd /home/molly/project/hftbacktest
RUN_ID="<填 AWS 上的 RUN_ID>"
mkdir -p local_live_analysis/$RUN_ID/raw_market_data
mkdir -p local_live_analysis/$RUN_ID/logs
```

从 AWS 拉取：

```bash
scp admin@awsserver1:/home/admin/hft_live/runs/$RUN_ID/config_live.toml local_live_analysis/$RUN_ID/
scp admin@awsserver1:/home/admin/hft_live/runs/$RUN_ID/binancefutures.toml local_live_analysis/$RUN_ID/
scp admin@awsserver1:/home/admin/hft_live/runs/$RUN_ID/logs/*.log local_live_analysis/$RUN_ID/logs/
scp admin@awsserver1:/home/admin/hft_live/runs/$RUN_ID/data/*.gz local_live_analysis/$RUN_ID/raw_market_data/
scp admin@awsserver1:/home/admin/hft_live/runs/$RUN_ID/output/audit_live*.csv local_live_analysis/$RUN_ID/ || true
scp admin@awsserver1:/home/admin/hft_live/hftbacktest/examples/binance_tick_mm/audit_live*.csv local_live_analysis/$RUN_ID/ || true
```

确认：

```bash
find local_live_analysis/$RUN_ID -maxdepth 3 -type f | sort
ls -lh local_live_analysis/$RUN_ID/raw_market_data/*.gz
```

### 验收

- 本地目录中存在 live audit CSV。
- 本地目录中存在 raw market gzip。
- 本地目录中存在 connector/live bot 日志。
- audit CSV 行数大于 0。
- gzip 文件已经位于本地：

```text
local_live_analysis/<RUN_ID>/raw_market_data/*.gz
```

后续禁止直接在 AWS 上做最终分析；AWS 只保留运行原始产物，本地 Ubuntu 负责转换、回测、对齐和归档。

## 4. 阶段 D：生成本地回放输入

### 4.1 生成订单延迟模型

```bash
cd /home/molly/project/hftbacktest
RUN_ID="<填 RUN_ID>"
AUDIT="$(ls local_live_analysis/$RUN_ID/audit_live*.csv | head -1)"

python examples/binance_tick_mm/latency_from_audit.py \
  --audit-csv "$AUDIT" \
  --output-npz "local_live_analysis/$RUN_ID/live_order_latency.npz" \
  --output-stats "local_live_analysis/$RUN_ID/live_order_latency_stats.json"
```

验收：

- `live_order_latency.npz` 存在。
- stats JSON 存在。
- rows > 0。
- entry/resp latency 分布合理。

### 4.2 转换 live raw market data

```bash
RAW_GZ="$(ls local_live_analysis/$RUN_ID/raw_market_data/*.gz | head -1)"

python examples/binance_tick_mm/pipeline_live_raw.py \
  --input-gz "$RAW_GZ" \
  --out-dir "local_live_analysis/$RUN_ID/out/live_raw" \
  --symbol BTCUSDT \
  --start-day "$(date -u +%Y-%m-%d)" \
  --end-day "$(date -u +%Y-%m-%d)"
```

如果 date 不对应实际 run 日期，应手动填入实盘日期。

验收：

- manifest 存在：

```text
local_live_analysis/$RUN_ID/out/live_raw/btcusdt/manifest_YYYY-MM-DD_to_YYYY-MM-DD.json
```

- NPZ 存在且 `data` 行数 > 0。
- 如果 gzip truncated，转换脚本应仍能处理完整行。

## 5. 阶段 E：生成 aligned backtest config

从 live config 复制一份 backtest config：

```bash
cp local_live_analysis/$RUN_ID/config_live.toml local_live_analysis/$RUN_ID/config_backtest_align.toml
```

手动或脚本修改：

```toml
[paths]
output_root = "/home/molly/project/hftbacktest/local_live_analysis/<RUN_ID>/out/backtest_align"

[latency]
order_latency_npz = "/home/molly/project/hftbacktest/local_live_analysis/<RUN_ID>/live_order_latency.npz"
latency_guard_ms = 5.0

[audit]
output_csv = "audit_bt_align.csv"
flush_every = 100

[summary]
enabled = true
output_json = "summary_align.json"
daily_csv = "daily_summary_align.csv"

[backtest]
window = "full_day"
wait_timeout_ns = 1000000
```

第一轮先使用 normal cadence：

```toml
[backtest_cadence]
mode = "fixed_interval"
enabled = false
min_decision_interval_ms = 0.0
audit_csv = ""
run_id = ""
ts_column = "ts_local"
tolerance_ms = 0.0
```

第二轮使用 audit replay：

```toml
[backtest_cadence]
mode = "audit_replay"
enabled = true
audit_csv = "/home/molly/project/hftbacktest/local_live_analysis/<RUN_ID>/<AUDIT_LIVE>.csv"
run_id = "<live run_id>"
ts_column = "ts_local"
tolerance_ms = 0.0
```

### 验收

- config 中所有绝对路径存在或父目录可创建。
- `order_latency_npz` 指向本 run 的 latency 文件。
- `output_root` 指向本 run 目录下，不覆盖旧 run。

## 6. 阶段 F：同窗口回测

### 6.1 获取 live 首尾时间

```bash
python - <<'PY'
import csv, sys
from pathlib import Path
audit = Path(sys.argv[1])
with audit.open(newline="") as f:
    rows = list(csv.DictReader(f))
ts = [int(float(r["ts_local"])) for r in rows if r.get("ts_local")]
print(min(ts), max(ts), len(ts))
PY "$AUDIT"
```

记录：

```text
SLICE_START=<first_ts_local>
SLICE_END=<last_ts_local>
LIVE_ROWS=<row_count>
```

### 6.2 normal cadence 回测

```bash
MANIFEST="$(ls local_live_analysis/$RUN_ID/out/live_raw/btcusdt/manifest_*.json | head -1)"

HFTBACKTEST_USE_LOCAL_PY=1 python examples/binance_tick_mm/backtest_tick_mm.py \
  --config "local_live_analysis/$RUN_ID/config_backtest_align.toml" \
  --manifest "$MANIFEST" \
  --window full_day \
  --slice-ts-local-start "$SLICE_START" \
  --slice-ts-local-end "$SLICE_END"
```

保存输出 JSON 到：

```text
local_live_analysis/$RUN_ID/backtest_normal_result.json
```

### 6.3 audit_replay cadence 回测

复制一份 config：

```bash
cp local_live_analysis/$RUN_ID/config_backtest_align.toml local_live_analysis/$RUN_ID/config_backtest_audit_replay.toml
```

修改 `[backtest_cadence]` 为 `audit_replay` 后运行：

```bash
HFTBACKTEST_USE_LOCAL_PY=1 python examples/binance_tick_mm/backtest_tick_mm.py \
  --config "local_live_analysis/$RUN_ID/config_backtest_audit_replay.toml" \
  --manifest "$MANIFEST" \
  --window full_day \
  --slice-ts-local-start "$SLICE_START" \
  --slice-ts-local-end "$SLICE_END"
```

保存输出 JSON 到：

```text
local_live_analysis/$RUN_ID/backtest_audit_replay_result.json
```

### 验收

normal cadence：

- 回测完成。
- audit CSV 存在。
- summary JSON 存在。
- `rows > 0`。

audit replay：

- 回测完成。
- `audit_replay_scheduled_count` 接近 live rows。
- `audit_replay_consumed_count / audit_replay_scheduled_count >= 0.99`。
- `audit_replay_unconsumed_count` 尽量 <= 1。

## 7. 阶段 G：生成对齐报告

### normal cadence report

```bash
BT_NORMAL="local_live_analysis/$RUN_ID/out/backtest_align/audit_bt_align.csv"

python examples/binance_tick_mm/compare_audit.py \
  --bt "$BT_NORMAL" \
  --live "$AUDIT" \
  --out "local_live_analysis/$RUN_ID/alignment_report_normal.json"
```

### audit replay report

如果 audit replay 写到同一个输出文件，先调整 config 输出名避免覆盖。建议：

```toml
[audit]
output_csv = "audit_bt_audit_replay.csv"
```

然后：

```bash
BT_REPLAY="local_live_analysis/$RUN_ID/out/backtest_align/audit_bt_audit_replay.csv"

python examples/binance_tick_mm/compare_audit.py \
  --bt "$BT_REPLAY" \
  --live "$AUDIT" \
  --out "local_live_analysis/$RUN_ID/alignment_report_audit_replay.json"
```

提取关键指标：

```bash
python - <<'PY'
import json, sys
from pathlib import Path
for raw in sys.argv[1:]:
    p = Path(raw)
    r = json.loads(p.read_text())
    print("==", p)
    print("bt_rows", r["bt_summary"]["rows"])
    print("live_rows", r["live_summary"]["rows"])
    print("common_rows", r["alignment"]["common_rows"])
    print("action_match_rate", r["alignment"]["action_match_rate"])
    print("reject_reason_match_rate", r["alignment"]["reject_reason_match_rate"])
    print("bt_drop_latency_rate", r["bt_summary"]["drop_latency_rate"])
    print("live_drop_latency_rate", r["live_summary"]["drop_latency_rate"])
    print("bt_drop_api_rate", r["bt_summary"]["drop_api_rate"])
    print("live_drop_api_rate", r["live_summary"]["drop_api_rate"])
    print("mae", r["alignment"]["mae"])
    print("nearest_lag", r["cadence"]["nearest_lag_live_to_bt"])
PY \
local_live_analysis/$RUN_ID/alignment_report_normal.json \
local_live_analysis/$RUN_ID/alignment_report_audit_replay.json
```

## 8. 阶段 H：验收标准

### H1. 实盘数据验收

必须通过：

- `ORDER_TRADE_UPDATE` 日志存在。
- audit live 非空。
- raw market gzip 非空。
- live 结束时没有持续 open-order drift。
- position mismatch 没有 critical。

建议门槛：

- live rows > `50,000`。
- duration > `60min`。
- final position mismatch <= `0.001 BTC`。

### H2. 回测执行验收

必须通过：

- live raw NPZ 转换成功。
- latency NPZ 生成成功。
- normal cadence 回测完成。
- audit replay 回测完成。
- compare report 成功写出。

### H3. 对齐指标验收

短期合格线：

- audit replay rows 与 live rows 基本一致：
  - `consumed / scheduled >= 0.99`
- action match rate：
  - 合格：`>= 0.85`
  - 目标：`>= 0.90`
- reject reason match rate：
  - 必须显著高于旧基线 `0.51`
  - 初步目标：`>= 0.60`
- latency drop rate 差异：
  - 初步目标：绝对差 <= `0.05`
- API drop rate 差异：
  - 初步目标：绝对差 <= `0.05`
- fair/reservation MAE：
  - 应低于旧 Plan B 的约 `12`
  - 目标逐步靠近 tick 级
- position MAE：
  - 应低于旧 Plan B 的约 `0.0018 BTC`
  - 不应出现系统性偏多或偏空

### H4. 是否进入参数优化的门槛

只有满足以下条件，才进入 sweep/walk-forward：

- open-order lifecycle 已稳定。
- audit replay 可稳定复现 live rows。
- action match rate 至少稳定在 `0.85+`。
- reject/latency/API 差异有清晰解释。
- position path 没有明显模型性偏差。

如果未满足，不做策略参数优化，继续校准执行模型。

## 9. 阶段 I：差异定位顺序

如果对齐报告不达标，按以下顺序定位。

### 9.1 Cadence

症状：

- normal rows 与 live rows 差很多。
- audit replay 明显改善 fair/reservation MAE。

处理：

- 保留 audit replay 作为复现上限。
- 不再简单扫 fixed interval。
- 分析 live `ts_local` delta 的 burst/gap 分布。

### 9.2 Latency / gating

症状：

- reject reason match 低。
- latency drop rate 差异大。

处理：

- 分开校准：
  - order latency model
  - latency guard signal
- 对比 live feed latency 与 backtest predicted entry latency 的触发差异。

### 9.3 API throttle

症状：

- API drop rate 差异大。
- action match 受 `api_interval_guard` 或 token bucket 影响。

处理：

- 调 `min_interval_ms`。
- 调 token bucket `capacity/refill_per_sec`。
- 检查 live 是否有 quote throttle 或额外 safety throttle 未被 backtest 消费。

### 9.4 Fill / queue

症状：

- action match 还可以，但 position MAE 大。
- 仓位方向有系统偏差。

处理：

- 用 `ORDER_TRADE_UPDATE` 对齐 fill path。
- 调 `power_prob_n`。
- 对比 `NoPartialFillExchange` 与 `PartialFillExchange`。

### 9.5 策略参数

只有前面几项合理后再调：

- `base_spread`
- `k_inv`
- `k_pos`
- `w_imb`
- `impact`

## 10. 阶段 J：进入优化

复现层达标后：

### Walk-forward

```bash
python examples/binance_tick_mm/walk_forward.py \
  --config ./examples/binance_tick_mm/config.toml \
  --target amdserver \
  --start-day YYYY-MM-DD \
  --end-day YYYY-MM-DD \
  --train-days 7 \
  --test-days 1 \
  --window full_day
```

### Parameter sweep

```bash
python examples/binance_tick_mm/sweep_backtest.py \
  --base-config ./examples/binance_tick_mm/config.toml \
  --manifest /path/to/manifest.json \
  --grid /path/to/sweep.toml \
  --workers 8 \
  --window full_day \
  --out /path/to/sweeps
```

大规模回测建议：

```toml
[audit]
mode = "off"

[summary]
enabled = true
output_json = "summary.json"
daily_csv = "daily_summary.csv"
```

## 11. 阶段 K：本地归档

### 目标

每次 run 在本地分析完成后，生成不可变归档，包含原始实盘数据、配置、日志、转换产物、回测结果、对齐报告和人工总结。归档后即使 AWS 上旧 run 被清理，本地仍可完整复现。

### 操作

在本地 Ubuntu 执行：

```bash
cd /home/molly/project/hftbacktest
RUN_ID="<填 RUN_ID>"
ARCHIVE_ROOT="local_live_analysis/archive"
mkdir -p "$ARCHIVE_ROOT"
```

确认本 run 总结文件已写好：

```bash
test -f local_live_analysis/$RUN_ID/live_alignment_summary.md
```

生成文件清单和 checksum：

```bash
find local_live_analysis/$RUN_ID -type f | sort > local_live_analysis/$RUN_ID/FILE_MANIFEST.txt
sha256sum $(find local_live_analysis/$RUN_ID -type f | sort) > local_live_analysis/$RUN_ID/SHA256SUMS.txt
```

创建压缩归档：

```bash
tar -czf "$ARCHIVE_ROOT/${RUN_ID}.tar.gz" -C local_live_analysis "$RUN_ID"
sha256sum "$ARCHIVE_ROOT/${RUN_ID}.tar.gz" > "$ARCHIVE_ROOT/${RUN_ID}.tar.gz.sha256"
```

可选：为归档写一个索引条目：

```bash
cat >> "$ARCHIVE_ROOT/INDEX.md" <<EOF

## $RUN_ID

- Archive: \`${RUN_ID}.tar.gz\`
- Checksum: \`${RUN_ID}.tar.gz.sha256\`
- Summary: \`../$RUN_ID/live_alignment_summary.md\`
- Raw gzip: \`../$RUN_ID/raw_market_data/\`
- Created at: $(date -Iseconds)
EOF
```

### 验收

- `local_live_analysis/archive/<RUN_ID>.tar.gz` 存在且非空。
- `local_live_analysis/archive/<RUN_ID>.tar.gz.sha256` 存在。
- `local_live_analysis/<RUN_ID>/FILE_MANIFEST.txt` 存在。
- `local_live_analysis/<RUN_ID>/SHA256SUMS.txt` 存在。
- 归档中包含：
  - raw gzip
  - live audit
  - config
  - logs
  - latency NPZ
  - live raw NPZ/manifest
  - backtest audit/summary
  - alignment reports
  - `live_alignment_summary.md`

快速验证归档：

```bash
tar -tzf local_live_analysis/archive/${RUN_ID}.tar.gz | rg "raw_market_data|audit_live|alignment_report|live_alignment_summary"
sha256sum -c local_live_analysis/archive/${RUN_ID}.tar.gz.sha256
```

### AWS 清理建议

归档验收通过前，不清理 AWS 上对应 run 目录。

归档验收通过后，才可以在 AWS 上按需清理旧 raw gzip 或旧 run：

```bash
ssh admin@awsserver1
du -sh /home/admin/hft_live/runs/<RUN_ID>
# 确认本地归档存在后再清理
```

## 12. 本轮交付物清单

一次完整执行后，本地应至少有：

```text
local_live_analysis/<RUN_ID>/
├── config_live.toml
├── config_backtest_align.toml
├── config_backtest_audit_replay.toml
├── binancefutures.toml
├── audit_live*.csv
├── live_order_latency.npz
├── live_order_latency_stats.json
├── logs/
│   ├── collector.log
│   ├── connector.log
│   └── live_bot.log
├── raw_market_data/
│   └── *.gz
├── out/
│   ├── live_raw/
│   └── backtest_align/
├── backtest_normal_result.json
├── backtest_audit_replay_result.json
├── alignment_report_normal.json
└── alignment_report_audit_replay.json
```

最后更新一份人工总结：

```text
local_live_analysis/<RUN_ID>/live_alignment_summary.md
```

内容包括：

- run 时间和配置。
- 实盘安全状态。
- ORDER_TRADE_UPDATE 验证结果。
- normal/audit_replay 对齐指标。
- 是否达到优化门槛。
- 下一步具体校准项。

归档后还应有：

```text
local_live_analysis/archive/
├── <RUN_ID>.tar.gz
├── <RUN_ID>.tar.gz.sha256
└── INDEX.md
```
