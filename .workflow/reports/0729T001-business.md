# 0729T001 业务执行回报

执行线程：
- 业务执行线程

任务ID：
- 0729T001

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `latency-probe/ec2_spread_hunt.py`
- `latency-probe/ec2_hunt.py`
- `latency-probe/test_ec2_hunt.py`
- `docs/ec2_latency_hunt_protocol.md`
- `docs/evidence/ec2_latency_hunt_20260729/`
- `.workflow/tasks/0729T001.md`
- `.workflow/reports/0729T001-business.md`
- controller workflow files

action：
- 修正普通 Region 的 placement 约束：使用一个
  `strategy=spread / spread-level=rack` group，而不是仅 Outposts 支持的
  host level。
- 在 `ap-northeast-1c` 分两批启动 2 + 5 台完全相同的
  `c6in.xlarge`；AWS placement receipt 验证 7/7 同 AZ、同机型、同 group。
- 复用 accepted Linux binary SHA `ba2fbaa4...3403`，先执行 2 台 60 秒
  canary，再执行 7 台 900 秒 Binance Futures BTCUSDT 与 Hyperliquid
  BTC fast-L2 正式窗口。
- 启动前冻结综合排名权重和 tie break，结果产生后未修改。
- 7/7 通过后保留综合第一的实际实例 `i-0a962e47210528526`，写 winner
  tags 并启用 API termination protection。
- 终止 6 台落选者、删除临时 bucket，保留 winner 所在 spread group。
- `awsserver1 / i-02c64c088f311cbc1` 全程保持 running，未 stop、reboot、
  terminate 或修改。

verify：
- Python focused tests：`22 passed`。
- Rust focused test：`15 passed`；clippy 通过。
- shell syntax、Python compile 和 worktree diff check：通过。
- 独立 QA 后发现当时 staged diff check 有两处 EOF 空行；该项由
  `0729T002` 离线修复，本报告不再把 staged diff 声明为已通过。
- Canary setup/smoke/upload/download：`2/2`。
- Full SSM commands：`7/7 Success`。
- Feed / local eligibility：`7/7`。
- before/after complete clock bounds：`412.394-655.988us`，全部
  `<=750us`。
- remote raw-to-summary rebuild：`14/14`。
- independent local raw-to-summary rebuild：`14/14`。
- Binance traces：每台 `437,121-439,801`。
- Hyperliquid fast-L2：每台 `1,662/900s`。
- AWS independent final query：
  - running servers exactly `2`；
  - `awsserver1` unchanged and running；
  - retained c6in winner running；
  - six losers terminated；
  - winner termination protection true；
  - temporary bucket explicit `404`；
  - one rack-spread group retained with winner。

done：
- 冻结综合第一：
  `candidate-07 / i-0a962e47210528526 / score 31`。
- Binance feed P50 winner、Binance local P99 winner 和 benchmark P99
  winner均为 retained candidate-07。
- Binance feed P99 单项 winner 是已终止的 candidate-05：
  `2.951ms`；retained balanced winner 为 `12.767ms`。二者未混写。
- 同机型 placement 差异得到实证：Binance feed P99
  `2.951-14.220ms`，约 `4.82x`；本地路径差异约 `1.08-1.24x`。
- 终态达到用户要求：只保留 `awsserver1 + 本轮真实 c6in 胜者`。

blockers：
- 无任务内执行阻塞。
- 本任务仍不证明真实 order ACK、fill、PnL 或 8 小时生产稳定性。
- 胜者在 burn-in 和迁移验收前不得 stop/start，也不得释放
  `awsserver1`。

commit：
- 无

提交信息：
- 无
