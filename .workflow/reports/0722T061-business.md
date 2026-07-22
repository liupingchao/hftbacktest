# 0722T061 Business Report

执行线程：
- 总控 / 业务执行线程

任务ID：
- 0722T061

状态：
- 待验收

更新时间：
- 2026-07-22 15:48 Asia/Shanghai

是否进行QA验收：
- 是

QA说明：
- 无

files：
- `examples/hyperliquid/cross_exchange_basis_regression_acceptance.py`
- `examples/hyperliquid/test_cross_exchange_basis_regression_acceptance.py`
- `local_live_analysis/cross_exchange_basis_regression_acceptance_0722T061/`
- `.workflow/tasks/0722T061.md`
- `.workflow/reports/0722T061-business.md`
- `task_plan.md`
- `progress.md`
- `findings.md`

action：
- 实现 fixed-model basis regression acceptance runner。
- 只消费 `0627T001` QA-accepted public rows，保留 nominal/effective 1000ms
  row gate。
- 三个固定模型按 leave-one-window-out 评估；normalization 及 OLS
  coefficient 只使用 train fold。
- 输出 standardized/raw coefficient、held-out prediction/performance、
  aggregate comparison、source-age/basis/regime stability、split/leakage
  manifest 和 public-shadow-only frozen contract。
- 生成 artifact 路径均为 repo-relative，不写本机绝对路径。

verify：
- Synthetic acceptance/reject/context-only/boundary/leakage/determinism/
  portability：
  `7 passed in 0.13s`。
- Signal/kernel/shadow related focused：
  `25 passed in 0.35s`。
- Official package deterministic rerun：
  所有 artifact SHA-256 完全一致。
- Official contract/boundary/leakage `jq` gates：全部通过。
- Full Hyperliquid：
  `1267 passed, 2 skipped in 56.78s`。
- Conda `py_compile`、`git diff --check` 和 absolute-path scan 通过。

done：
- Official recommendation：
  `accept_basis_regression_for_shadow`。
- Valid OOS rows：
  `10,704` across `3` windows。
- Aggregate baseline vs combined：
  - direction hit：`0.7150910668` -> `0.7571552472`
  - RMSE ticks：`29.0598046581` -> `26.9460638414`
  - correlation：`0.1239906948` -> `0.4379254498`
  - mean signed move ticks：`3.9106875934` -> `6.749813154`
- Combined fold raw basis slopes：
  `0.1959017750 / 0.2483477961 / 0.2575285232`，方向一致。
- Frozen full-data raw basis slope：
  `0.2228155507` future-HL ticks per basis tick。
- 强制 warnings：
  - combined aggregate MAE `18.3390402034` 高于 baseline
    `12.0231909201`
  - `xemm_0627_t001_hlfast_utc17_c` 同时在 direction/RMSE 落后 baseline
  - raw intercept range `11.1676762654` ticks
  - held-out prediction mean range `27.7840120699` ticks
  - only three accepted windows
  - Binance USD-M `BTCUSDT` vs Hyperliquid `BTC` contract-basis caveat
- Contract 只授权后续 production-equivalent public shadow，不授权 live、
  execution PnL、promotion 或 default-on。

blockers：
- 独立 QA 验收。

commit：
- `8cd50407edc116e40f92bef7bd5dd182d6e65725`

提交信息：
- `Implement basis regression alpha acceptance`
