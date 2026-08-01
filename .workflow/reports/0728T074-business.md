# 0728T074 业务执行回报

执行线程：
- 业务执行线程

任务ID：
- 0728T074

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
- `docs/evidence/ec2_latency_hunt_20260728/control_canary.json`
- `docs/evidence/ec2_latency_hunt_20260728/execution_manifest.json`
- `docs/evidence/ec2_latency_hunt_20260728/README.md`
- workflow controller files

action：
- S3 bucket 只有明确 `404/NoSuchBucket` 才按不存在处理；403、权限、
  限流、网络或未知错误进入 cleanup failure。
- bucket tag discovery 只忽略 NoSuchTagSet/明确不存在，其他错误抛出。
- setup SSM Success 后解析 stdout clock JSON，验证 Normal、gate=pass 和
  bound<=limit，并写 `setup_clock_receipts_verified`。
- remote setup 的 clock receipt `cat` 失败立即非零。
- remote upload 抽成可执行函数，PUT 失败返回非零并保留 JSON receipt。
- 增加真实 PUT failure、HeadBucket 403、SIGTERM/finally、setup receipt
  缺失/gate fail 和 failed clock journal regressions。
- 用最终代码运行 c7i.xlarge、m7i.xlarge 两台 60 秒 Canary。

verify：
- Python：`16 passed`。
- Rust fmt/test/clippy：通过，Rust `15 passed`。
- final setup SSM：`2/2 Success`，receipt gate event `2/2`。
- final run SSM：`2/2 Success`。
- upload receipt / S3 size / download SHA：均 `2/2`。
- raw-to-summary rebuild：`4/4`。
- Binance traces：`14389`、`18474`；Hyperliquid fast-L2：
  `110`、`109`。
- cleanup state：`verified=true`、errors `[]`、post_state 全空。
- 独立 AWS 查询：active instances `[]`、placement groups `[]`、
  temporary bucket 为明确 `404 Not Found`。

done：
- T073 QA 的 S3 403 fail-open、setup receipt gate 和 fault-test coverage
  三项缺陷均已修复。
- T072 accepted 10-host full data 和 deterministic build evidence 保持
  不变。
- Goal 2 再次具备可进入独立 QA 的完整数据层和 fail-closed 控制面。

blockers：
- 无任务内阻塞。
- Hyperliquid feed-network 仍不用于实例排名；不证明订单 ACK/fill/PnL。

commit：
- 无

提交信息：
- 无
