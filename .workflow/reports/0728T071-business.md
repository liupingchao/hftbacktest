# 0728T071 业务执行回报

执行线程：
- 业务执行线程

任务ID：
- 0728T071

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `latency-probe/src/main.rs`
- `latency-probe/ec2_hunt.py`
- `latency-probe/test_ec2_hunt.py`
- `latency-probe/ec2_hunt_remote.sh`
- `docs/ec2_latency_hunt_protocol.md`
- `docs/evidence/ec2_latency_hunt_20260728/`
- `.workflow/tasks/0728T071.md`
- `.workflow/reports/0728T071-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 在启动前审查 AWS sample `trading-latency-benchmark` commit `8dad243`，
  纳入独立 cluster placement group、warmup、Chrony gate、host/ENA
  metadata、并发执行和自动回收边界。
- 新增 `--warmup-messages`；warmup 完成后才启动完整 duration window。
- 用一次性 builder 构建 Linux x86_64 release binary；回收 binary 与最小
  lockfile 后终止 builder。
- 发现基础 AMI 未包含可复现 SSM 注册状态，改为所有实例 user-data 显式
  安装并启动 SSM Agent。
- 因本地没有 `key1` RSA 私钥，使用私有 task-scoped S3 + SSM 分发，不
  开放 SSH ingress。
- 2 台 canary smoke 通过后扩至 10 台 x86_64 `xlarge` 候选，每台独立
  cluster placement group。
- 每台双 venue 各 100 条 warmup 后并发采集 900 秒；保存 raw、summary、
  benchmark、Chrony、CPU/ENA/offload 和哈希证据。
- 10 个结果包全部回收后终止全部实例，删除 placement groups 和临时桶。

verify：
- Rust fmt/test/clippy：通过，`14 passed`。
- hunting analyzer pytest：`2 passed`。
- canary：`2/2` 成功，双 venue trace 非零、完整、零 drop。
- full SSM command：`10 success / 0 failed`。
- 候选 validation：`10/10 eligible`。
- 每台 duration `900s`、warmup `100`、clock gate pass、offset `0-2us`。
- 远端 raw summary rebuild：`20/20`。
- 本机独立 raw summary rebuild：`20/20`。
- T071 active instances：`[]`。
- T071 placement groups：`[]`。
- 临时 S3 bucket：已删除。

done：
- 推荐 `c6in.xlarge` 作为第一胜出实例：Binance feed P50 第3、P99 第2，
  Binance tick-to-wire P99 第1，Hyperliquid tick-to-wire P99 第2。
- `c5n.xlarge` 为网络 tail control：Binance feed P99 第1，但本地 pipeline
  慢于 `c6in.xlarge`。
- 证明候选间存在显著差异：Binance feed P99 `3.575-75.435ms`，Binance
  tick-to-wire P99 `12.836-34.109us`。
- Hyperliquid exchange timestamp 与 frame receipt 相差约数百毫秒，当前
  `feed_network` 不进入推荐排名；monotonic tick-to-wire 仍有效。

blockers：
- 无任务内执行阻塞。
- 本任务不包含 tuned-vs-untuned A/B，也不证明订单 ACK/fill/PnL。

commit：
- 无

提交信息：
- 无
