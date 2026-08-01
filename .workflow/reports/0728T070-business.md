# 0728T070 业务执行回报

执行线程：
- 业务执行线程

任务ID：
- 0728T070

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `Cargo.toml`
- `latency-probe/Cargo.toml`
- `latency-probe/src/lib.rs`
- `latency-probe/src/main.rs`
- `docs/latency_observability_contract.md`
- `docs/evidence/latency_probe_20260728/`
- `.workflow/tasks/0728T070.md`
- `.workflow/reports/0728T070-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 新建 workspace crate `latency-probe`，定义 15 个固定生命周期阶段、两类
  时钟域和 14 个标准 interval。
- 原始 trace 使用固定 timestamp arrays 和唯一 trace id；summary 统计
  percentile、missing、invalid、duplicate、incomplete、out-of-order 和
  bounded-queue drop。
- 公共探针支持 Binance Futures `bookTicker` 与 Hyperliquid `bbo`。
- 实现 BBO apply、signal、decision、JSON intent encode 和本机 UDP socket
  write 的完整本地链路；不发送真实订单。
- 原始 trace 通过 bounded `try_send` 交给独立 writer thread，磁盘序列化
  不进入 tick-to-wire。
- 新增 `synthetic`、`summarize`、`benchmark` 命令和可选
  `--connect-ip`、`--tls12-only` DNS/TLS/edge 诊断参数。
- `socket_receive`、ACK、resting、fill、cancel 阶段未伪造，在公共探针
  summary 中按 missing 计数。

verify：
- `cargo +1.93.0 fmt -p latency-probe -- --check`：通过；仓库
  `rustfmt.toml` 的 nightly-only 选项产生 warning，但目标 crate 无 diff。
- `cargo +1.93.0 clippy -p latency-probe --all-targets -- -D warnings`：通过。
- `cargo +1.93.0 test -p latency-probe`：`13 passed`。
- CLI `--help` / `public --help`：通过，四个命令和两个诊断开关可见。
- 1,000 条 synthetic NDJSON 经过独立 `summarize` 后 summary 字节完全一致；
  `1000/1000` complete，duplicate/out-of-order/drop 均为 `0`。
- 隔离 release recorder benchmark，1,000,000 个真实逐 trace 样本：
  `batch_size=1`，P50 `334 ns/trace`，P99 `500 ns/trace`，max
  `53042 ns/trace`，drop `0`，低于 `5 us` P99 门槛。
- Binance release public smoke，100 traces：`100/100` complete，drop `0`；
  tick-to-wire P50 `7.271 us`，P99 `66.089 us`。
- Hyperliquid release public smoke，100 traces：`100/100` complete，drop `0`；
  tick-to-wire P50 `42.521 us`，P99 `109.273 us`。
- Hyperliquid 默认 TLS 协商在本机透明代理中 reset；显式
  `--tls12-only` 后沿默认 DNS 路径实流采样通过。`--connect-ip` 也已单独
  验证会保留官方 hostname/SNI。
- 接受证据保存在 `docs/evidence/latency_probe_20260728/`：双交易所 raw
  NDJSON/summary、benchmark JSON 和 run manifest；manifest 包含命令、
  host/toolchain、DNS/CDN IP、source SHA-256 和 artifact SHA-256。
- 新增零有效 BBO fail-closed、queue full 和 writer disconnected 回归。
- `git diff --check`：通过。

done：
- Goal 1 的统一时延契约、可审计 raw trace、P50/P99 统计、完整性计数、
  bounded handoff 和两家交易所公共探针已经实现。
- 两家交易所均产生同 schema 的真实公共行情 trace。
- 当前 release 本地 probe/loopback 链路 P99 均低于 1 ms。
- 本任务没有执行 credential/private/account/order/cancel/remote/service
  操作，不支持 ACK/fill/PnL 或生产交易时延结论。

blockers：
- 无任务内阻塞。
- Binance `feed_network` 有 `45/100` 负样本，证明当前开发机墙钟未与交易所
  对齐；该指标在 NTP/PTP 证据建立前不能作为网络时延验收值。
- Connector socket receive、真实策略进程、exchange ACK/resting/fill/cancel
  和 awsserver1/EC2 hunting 属于后续正式任务。

commit：
- 无

提交信息：
- 无
