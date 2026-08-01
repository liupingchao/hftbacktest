# 0728T072 业务执行回报

执行线程：
- 业务执行线程

任务ID：
- 0728T072

状态：
- 待验收

是否进行QA验收：
- 是

files：
- `latency-probe/src/main.rs`
- `latency-probe/src/lib.rs`
- `latency-probe/ec2_hunt.py`
- `latency-probe/ec2_hunt_build.py`
- `latency-probe/ec2_hunt_orchestrator.py`
- `latency-probe/ec2_hunt_remote.sh`
- `latency-probe/ec2_hunt_user_data.sh`
- `latency-probe/test_ec2_hunt.py`
- `docs/ec2_latency_hunt_protocol.md`
- `docs/evidence/ec2_latency_hunt_20260728/`
- workflow controller files

action：
- 修复完整 Chrony error bound 门禁，并在 accepted window 前后分别实时
  保存 `chronyc tracking`。
- analyzer 区分 feed eligibility 与 monotonic local-pipeline eligibility。
- 新增原子 state journal、state+tag discovery、signal/finally cleanup 和
  regional SigV4 presigned transfer。
- Hyperliquid 从稀疏 `bbo` 切换到 live 路径一致的
  `l2Book fast=true`，从 top level 派生 BBO。
- 使用现有 awsserver1 构建宿主产出 Linux x86_64 binary。
- 三次失败尝试均 fail-closed 并自动清理；第四次完成 10 台正式采集。

verify：
- Python pytest：`4 passed`。
- Rust fmt/test/clippy：通过，Rust `15 passed`。
- full SSM：10 个正式命令全部 Success。
- `10/10` feed eligible，`10/10` local-pipeline eligible。
- before/after complete clock bound：`79.903-596.035us`，均低于 `750us`。
- 远端 raw rebuild：`20/20`。
- 本机独立 raw rebuild：`20/20`。
- archive SHA：`10/10`。
- Hyperliquid fast-L2：每台 `1653/900s`。
- T072 active instances：`[]`。
- T072 placement groups：`[]`。
- temporary bucket：已删除。

done：
- 推荐 `c6in.xlarge` 作为 balanced winner：Binance feed P99 第2、
  Binance local P99 第2、Hyperliquid local P99 第2。
- `m7i.xlarge` 为 network-tail winner：Binance feed P99
  `4.679ms`，矩阵第1。
- `m5zn.xlarge` 为 local-pipeline winner，但 Binance feed P99
  `101.820ms`，不作为首选部署实例。
- Feed P50 结果携带完整 clock bound；相邻小差异不做过度精确解释。

blockers：
- 无任务内执行阻塞。
- Hyperliquid feed-network 仍降级；本任务不证明订单 ACK/fill/PnL。

commit：
- 无

提交信息：
- 无
