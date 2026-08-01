# 0728T073 业务执行回报

执行线程：
- 业务执行线程

任务ID：
- 0728T073

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `latency-probe/ec2_hunt_orchestrator.py`
- `latency-probe/ec2_hunt_remote.sh`
- `latency-probe/ec2_hunt_build.py`
- `latency-probe/test_ec2_hunt.py`
- `docs/evidence/ec2_latency_hunt_20260728/build_source.tar.gz`
- `docs/evidence/ec2_latency_hunt_20260728/build_receipt.json`
- `docs/evidence/ec2_latency_hunt_20260728/control_canary.json`
- `docs/evidence/ec2_latency_hunt_20260728/execution_manifest.json`
- `docs/evidence/ec2_latency_hunt_20260728/README.md`
- workflow controller files

action：
- 远端 archive PUT 失败现在会返回非零，并输出实例、label、archive
  SHA-256、大小和上传状态的单行 JSON 回执。
- orchestrator 保存失败 SSM invocation 详情，对 S3 对象做 head/size
  校验，并在下载后核对 archive SHA-256。
- bucket 在 API 创建前写 planned state，创建后立即写 Task/RunId tags；
  cleanup 合并 state/tag discovery，检查所有删除返回值并做删除后复核。
- setup stdout 只保留完整 clock JSON，避免 apt 输出触发 SSM 截断。
- Linux 构建改为 deterministic source archive，保留 Cargo.toml、
  Cargo.lock、lib.rs、main.rs hashes 和 exact archive bytes。
- 在现有 awsserver1 重建出的 binary SHA 与 T072 正式 10 台运行完全
  相同。
- 最终使用 c7i.xlarge、m7i.xlarge 跑 2 台 60 秒控制路径 Canary。

verify：
- Python：`11 passed`。
- Rust fmt/test/clippy：通过，Rust `15 passed`。
- deterministic source archive 连续两次 SHA 相同：
  `9b448775...`.
- rebuilt Linux binary SHA：
  `ba2fbaa4...`，与 T072 accepted full run 相同。
- final Canary setup SSM：`2/2 Success`，stdout 为完整 clock JSON。
- final Canary run SSM：`2/2 Success`。
- upload receipt / S3 size / download SHA：均 `2/2`。
- raw-to-summary rebuild：`4/4`。
- Binance traces：`9397`、`9355`；Hyperliquid fast-L2：
  `108`、`109`。
- cleanup state：`verified=true`、errors `[]`、post_state 全空。
- 独立 AWS 查询：active instances `[]`、placement groups `[]`、
  temporary bucket 不存在。

done：
- T072 已接受的 10 台 900 秒数据无需重跑，且 exact build source 已
  闭环。
- T072 QA 的 upload、cleanup、bucket registration、clock receipt、
  build provenance 和故障测试缺陷均已修复。
- Goal 2 已具备 tracked、fail-closed、可重建的正式采集和控制面证据，
  等待独立 QA 裁决。

blockers：
- 无任务内阻塞。
- Hyperliquid feed-network 仍不用于实例排名；本任务仍不证明订单
  ACK/fill/PnL。

commit：
- 无

提交信息：
- 无
