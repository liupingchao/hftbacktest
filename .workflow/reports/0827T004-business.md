# 业务执行回报

执行线程：
- SKHYNIX Binance Phase Alignment Track A0-A4 业务线程

任务ID：
- 0827T004

状态：
- 待验收

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/skhynix_phase_alignment_track_a.py`
- `examples/hyperliquid/test_skhynix_phase_alignment_track_a.py`
- `docs/skhynix_binance_phase_alignment_track_a_outcome_blind_motif_discovery_plan_20260827.md`
- `local_live_analysis/skhynix_phase_alignment_track_a_0827T004/`
- `.workflow/tasks/0827T004.md`
- `.workflow/reports/0827T004-business.md`

action：
- 冻结 29 个正式历史 capture、9 个研究日期和 calibration/development/
  blocked-validation/no-refit-replay 角色。
- 逐消息重建 Binance top-5，校验 snapshot bridge 与 `U/u/pu` 连续性，
  生成 100ms causal reconstruction grid 和 200ms state observation grid。
- 使用 7 月 29 日 prior-session median/IQR 冻结归一化，构建保留 L1-L5
  空间面的多分辨率 outcome-blind feature bank。
- 比较 K=3/4/5/6 Student-t shifted-negative-binomial sticky HSMM、
  Gaussian HSMM、Student-t memoryless HMM、ridge VAR(1) 和 single-state
  Student-t null。
- 生成 maximal-run ledger、neutral transition grammar、duration、null、
  prototype diagnostic、causal online filter 和 prefix detector artifacts。
- 对 12.8s/25.6s/51.2s duration support 与 400ms observation grid 做
  稳健性 replay。
- 按用户追加要求执行分钟尺度低参数 robustness：1s grid、固定 13 维
  L1-L5 投影、K=2/3/4、shared emission scale、每状态 2 个负二项
  duration 参数，并比较 60s/120s/300s support。
- 修正 duration fitting：capture/chunk 首尾 run 按 right censor 处理，
  不再作为完整 dwell 进入负二项估计或分钟尾部计数。
- 全程未读取 future return、future midpoint/BBO、future volatility、
  markout、fill 或 PnL；未访问 private API、未下单、未采集新数据。

verify：
- `python -m pytest -q examples/hyperliquid/test_skhynix_phase_alignment_track_a.py`：
  `5 passed`。
- 合成 fixture 两次构建的压缩 NPZ SHA256 一致。
- 合成 `U/u/pu` gap 返回 fail-closed `depth_sequence_gap`。
- 修改未来输入不改变在线 HSMM filter 的历史 posterior prefix。
- 全量执行两次；第二次复用冻结 feature cache，模型选择和 primary
  classification 一致。
- 29 个 capture 均 `valid_fraction=1.0`，总计 1,292,945 grid rows，
  零 sequence gap。
- duration support 扩大到 51.2s 后 validation score 不变到小数点后
  9 位；400ms replay 的 validation/replay label agreement 约 0.996。
- 分钟尺度 K=3 HSMM 为 66 参数，300s support validation density
  `-37.1226`；39 参数 diagonal AR(1) 为 `-34.7476`，仍领先
  `2.3750`/row。
- 排除边界删失后，Q0/Q1/Q2 的完整 `>=300s` dwell 数分别为
  `0 / 8 / 11`，分钟尾部不具备全状态共同识别支持。
- `python -m py_compile`、`git diff --check`：通过。

done：
- A0 数据准入：通过，35.917 小时历史公共数据。
- A1 因果状态表示：通过。
- A2 neutral discrete-state gate：失败。K=6 Student-t HSMM validation
  density 为 `-149.570`，连续 ridge VAR(1) 为 `-127.672`。
- A3 grammar：仅诊断执行；因 upstream state gate 失败，不具备正式通过
  资格。离散 decoder 产生 417,628 个 maximal runs，多数 state median
  dwell 为一个 200ms step。
- A4 online recognition：失败。no-refit replay run recall 为 `0.365`，
  late-detection fraction 为 `0.529`。
- Primary classification：
  `continuous_state_no_discrete_phase_support`。
- Minute-scale robustness：
  `continuous_state_still_preferred_at_minute_scale`；不改变 primary
  classification。
- N/S/P/R semantic mapping、Track B 和 positive structural claim 均未授权。

blockers：
- 无执行阻塞。
- 科学结论存在已知证据上限：零 true prospective session；本任务只能给出
  历史 no-refit replay 结论。

commit：
- 7d26fa9a

提交信息：
- research: extend phase duration robustness to minutes
